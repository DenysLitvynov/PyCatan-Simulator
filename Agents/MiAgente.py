import random
import math
import json
import os
from Classes.Constants import MaterialConstants, BuildConstants, DevelopmentCardConstants
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface


# =============================================================================
# CROMOSOMA — 16 genes, todos en rango [0.0, 1.0]
#
# [0]  w_city       — preferencia por construir ciudad
# [1]  w_town       — preferencia por construir pueblo
# [2]  w_road       — preferencia por construir carretera
# [3]  w_card       — preferencia por comprar carta de desarrollo
#
# [4]  w_prob       — peso de la probabilidad del dado al evaluar nodos iniciales
# [5]  w_resource   — peso del tipo de recurso al evaluar nodos iniciales
# [6]  w_coastal    — penalización por nodos costeros en colocación inicial
#
# [7]  w_thief_prob     — atacar terrenos con dados altos (6/8) del enemigo
# [8]  w_thief_vp       — atacar al jugador con más puntos de victoria
# [9]  w_thief_noself   — evitar terrenos donde yo tengo pueblos/ciudades
#
# [10] w_trade_bank     — umbral de exceso para comerciar con banco (4:1)
# [11] w_trade_offer    — agresividad al proponer trades a jugadores
# [12] w_trade_accept   — umbral para aceptar ofertas recibidas
#
# [13] w_discard    — qué recursos proteger más al descartar (alto = protege mineral)
#
# [14] w_knight     — umbral para jugar carta caballero al inicio del turno
# [15] w_buy_card   — preferencia por comprar carta vs construir (modifica w_card)
# =============================================================================

CHROMOSOME_SIZE = 16


def random_chromosome():
    """Genera un cromosoma aleatorio con valores entre 0 y 1."""
    return [random.random() for _ in range(CHROMOSOME_SIZE)]


class MiAgente(AgentInterface):
    """
    Agente entrenado con algoritmo genético.
    Todas las decisiones están parametrizadas por un cromosoma de 16 genes.
    """

    def __init__(self, agent_id, cromosoma=None):
        super().__init__(agent_id)
        
        if cromosoma is not None:
            self.cromosoma = cromosoma
        else:
            # Intenta cargar el mejor cromosoma entrenado
            json_path = os.path.join(os.path.dirname(__file__), '..', 'best_chromosome.json')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    data = json.load(f)
                self.cromosoma = data['best_chromosome']
            else:
                # Fallback: pesos neutros
                self.cromosoma = [0.5] * CHROMOSOME_SIZE
        
        self._parse_chromosome()
        self.town_count = 0
        
    
    def _parse_chromosome(self):
        """Asigna cada gen a una variable con nombre legible."""
        c = self.cromosoma
        # Construcción
        self.w_city     = c[0]
        self.w_town     = c[1]
        self.w_road     = c[2]
        self.w_card     = c[3]
        # Posición inicial
        self.w_prob     = c[4]
        self.w_resource = c[5]
        self.w_coastal  = c[6]
        # Ladrón
        self.w_thief_prob   = c[7]
        self.w_thief_vp     = c[8]
        self.w_thief_noself = c[9]
        # Comercio
        self.w_trade_bank   = c[10]
        self.w_trade_offer  = c[11]
        self.w_trade_accept = c[12]
        # Descarte
        self.w_discard = c[13]
        # Cartas
        self.w_knight   = c[14]
        self.w_buy_card = c[15]

    # =========================================================================
    # FUNCIONES DE EVALUACIÓN — el corazón del agente
    # Estas funciones convierten los genes en puntuaciones numéricas
    # =========================================================================

    def _score_node(self, node_id):
        """
        Puntúa un nodo del tablero para decidir dónde construir/colocarse.
        Usa w_prob (probabilidad del dado) y w_resource (tipo de recurso).
        Penaliza nodos costeros con w_coastal.
        """
        node = self.board.nodes[node_id]
        score = 0.0

        # --- Puntuación por probabilidad de los hexágonos adyacentes ---
        for terrain_id in node['contacting_terrain']:
            terrain = self.board.terrain[terrain_id]
            prob = terrain['probability']

            # Los dados más probables son 6 y 8 (5 formas de sacarlos)
            # Los menos probables son 2 y 12 (1 forma de sacarlos)
            if prob in [6, 8]:
                prob_score = 3.0
            elif prob in [5, 9]:
                prob_score = 2.0
            elif prob in [4, 10]:
                prob_score = 1.5
            elif prob in [3, 11]:
                prob_score = 0.5
            else:  # 2, 12
                prob_score = 0.1

            score += self.w_prob * prob_score

        # --- Puntuación por tipo de recurso ---
        for terrain_id in node['contacting_terrain']:
            terrain = self.board.terrain[terrain_id]
            rtype = terrain['terrain_type']

            # Mineral (1) y cereal (0) son los más valiosos para ciudades y cartas
            # Madera (3) y arcilla (2) para carreteras y pueblos al inicio
            # Lana (4) es menos valiosa en general
            # Desierto (-1) no produce nada
            if rtype == 1:    # mineral
                res_score = 3.0
            elif rtype == 0:  # cereal
                res_score = 2.5
            elif rtype == 3:  # madera
                res_score = 2.0
            elif rtype == 2:  # arcilla
                res_score = 2.0
            elif rtype == 4:  # lana
                res_score = 1.0
            else:             # desierto
                res_score = -5.0

            score += self.w_resource * res_score

        # --- Penalización por nodo costero (menos hexágonos adyacentes) ---
        # Un nodo interior toca 3 hexágonos, uno costero toca 1 o 2
        n_terrains = len(node['contacting_terrain'])
        if n_terrains < 3:
            score -= self.w_coastal * 5.0

        return score

    def _score_terrain_for_thief(self, terrain_id, vp_dict):
        """
        Puntúa un terreno para decidir dónde colocar el ladrón.
        Queremos terrenos con dado alto del enemigo, con el líder,
        y que no sean nuestros.
        vp_dict: diccionario {player_id: victory_points}
        """
        terrain = self.board.terrain[terrain_id]
        score = 0.0

        # No podemos dejar el ladrón donde ya está
        if terrain['has_thief']:
            return -9999.0

        # --- Puntuación por probabilidad del dado ---
        prob = terrain['probability']
        if prob in [6, 8]:
            prob_score = 3.0
        elif prob in [5, 9]:
            prob_score = 2.0
        elif prob in [4, 10]:
            prob_score = 1.0
        else:
            prob_score = 0.0
        score += self.w_thief_prob * prob_score

        # --- Puntuación por puntos de victoria del enemigo adyacente ---
        enemy_vp = 0
        i_am_here = False
        for node_id in terrain['contacting_nodes']:
            player = self.board.nodes[node_id]['player']
            if player == self.id:
                i_am_here = True
            elif player != -1:
                enemy_vp = max(enemy_vp, vp_dict.get(player, 0))

        score += self.w_thief_vp * enemy_vp

        # --- Penalización fuerte si yo tengo un pueblo/ciudad ahí ---
        if i_am_here:
            score -= self.w_thief_noself * 20.0

        return score

    def _get_discard_priority(self):
        """
        Devuelve el orden de materiales a descartar primero.
        w_discard alto = protege mineral y cereal (estrategia ciudad)
        w_discard bajo = protege madera y arcilla (estrategia expansión)
        """
        if self.w_discard > 0.5:
            # Protegemos mineral y cereal → descartamos lana, madera, arcilla primero
            return [
                MaterialConstants.WOOL,
                MaterialConstants.WOOD,
                MaterialConstants.CLAY,
                MaterialConstants.CEREAL,
                MaterialConstants.MINERAL,
            ]
        else:
            # Protegemos madera y arcilla → descartamos mineral, lana, cereal primero
            return [
                MaterialConstants.MINERAL,
                MaterialConstants.WOOL,
                MaterialConstants.CEREAL,
                MaterialConstants.CLAY,
                MaterialConstants.WOOD,
            ]

    # =========================================================================
    # TRIGGERS — las funciones que llama el simulador
    # =========================================================================

    def on_game_start(self, board_instance):
        """
        Colocación inicial: elegimos el mejor nodo usando _score_node.
        Se llama dos veces (ida y vuelta del setup).
        """
        self.board = board_instance
        possibilities = self.board.valid_starting_nodes()

        if not possibilities:
            # Fallback: cualquier nodo válido
            node_id = 0
            return node_id, self.board.nodes[node_id]['adjacent'][0]

        # Puntuamos todos los nodos disponibles y elegimos el mejor
        best_node = max(possibilities, key=lambda n: self._score_node(n))

        # La carretera: elegimos el adyacente que tenga mejor puntuación
        adjacent = self.board.nodes[best_node]['adjacent']
        # Filtramos solo adyacentes que sean nodos válidos (dentro del tablero)
        valid_adjacent = [n for n in adjacent if 0 <= n < len(self.board.nodes)]
        best_road = max(valid_adjacent, key=lambda n: self._score_node(n))

        self.town_count += 1
        return best_node, best_road

    def on_turn_start(self):
        """
        Antes de tirar dados: podemos jugar carta de caballero.
        w_knight controla cuándo la jugamos:
        - alto (>0.7): solo jugamos si el ladrón está en nuestros terrenos
        - medio (0.3-0.7): jugamos siempre que tengamos caballero
        - bajo (<0.3): nunca jugamos caballero aquí (lo guardamos)
        """
        if not self.development_cards_hand.hand:
            return None

        # Buscamos carta de caballero
        for i, card in enumerate(self.development_cards_hand.hand):
            if card.type == DevelopmentCardConstants.KNIGHT:
                if self.w_knight < 0.3:
                    # Estrategia conservadora: no jugamos aquí
                    return None
                elif self.w_knight < 0.7:
                    # Estrategia media: siempre jugamos si tenemos
                    return self.development_cards_hand.select_card(i)
                else:
                    # Estrategia agresiva: solo si el ladrón nos afecta
                    for terrain in self.board.terrain:
                        if terrain['has_thief']:
                            for node_id in terrain['contacting_nodes']:
                                if self.board.nodes[node_id]['player'] == self.id:
                                    return self.development_cards_hand.select_card(i)
        return None

    def on_having_more_than_7_materials_when_thief_is_called(self):
        """
        Descartamos la mitad de nuestros recursos.
        El orden de descarte depende de w_discard.
        """
        priority = self._get_discard_priority()
        target = math.floor(self.hand.get_total() / 2)
        discarded = 0

        while discarded < target:
            for material in priority:
                if self.hand.resources.get_from_id(material) > 0:
                    self.hand.remove_material(material, 1)
                    discarded += 1
                    break

        return self.hand

    def on_moving_thief(self):
        """
        Movemos el ladrón al terreno con mayor puntuación.
        Usamos _score_terrain_for_thief con los puntos de victoria actuales.
        """
        # Intentamos obtener los puntos de victoria actuales de cada jugador
        # Como no tenemos acceso directo, usamos el número de pueblos+ciudades como proxy
        vp_proxy = {}
        for node in self.board.nodes:
            player = node['player']
            if player != -1:
                vp_proxy[player] = vp_proxy.get(player, 0) + (2 if node['has_city'] else 1)

        # Puntuamos todos los terrenos
        best_terrain = None
        best_score = -9999.0
        for terrain in self.board.terrain:
            score = self._score_terrain_for_thief(terrain['id'], vp_proxy)
            if score > best_score:
                best_score = score
                best_terrain = terrain['id']

        # Buscamos un enemigo adyacente al terreno elegido para robarle
        best_player = -1
        if best_terrain is not None:
            # Preferimos robar al jugador con más puntos
            best_enemy_vp = -1
            for node_id in self.board.terrain[best_terrain]['contacting_nodes']:
                player = self.board.nodes[node_id]['player']
                if player != -1 and player != self.id:
                    enemy_vp = vp_proxy.get(player, 0)
                    if enemy_vp > best_enemy_vp:
                        best_enemy_vp = enemy_vp
                        best_player = player

        return {'terrain': best_terrain if best_terrain is not None else 0,
                'player': best_player}

    def on_commerce_phase(self, board_instance=None):
        """
        Fase de comercio.
        - Si tenemos exceso de algún recurso (controlado por w_trade_bank), comerciamos con banco
        - Si w_trade_offer es alto, proponemos trades a jugadores
        """
        self.board = board_instance if board_instance else self.board

        # --- Umbral de exceso para comercio con banco ---
        # w_trade_bank en [0,1] → umbral de 3 a 6 unidades de exceso
        bank_threshold = 3 + int(self.w_trade_bank * 3)  # entre 3 y 6

        # Detectamos qué material nos falta para el objetivo más inmediato
        needed_material = self._get_most_needed_material()

        # Comercio con banco: si tenemos demasiado de algo, lo cambiamos
        for material_id in [MaterialConstants.CEREAL, MaterialConstants.MINERAL,
                             MaterialConstants.CLAY, MaterialConstants.WOOD,
                             MaterialConstants.WOOL]:
            amount = self.hand.resources.get_from_id(material_id)
            if amount >= bank_threshold and material_id != needed_material:
                return {'gives': material_id, 'receives': needed_material}

        # Propuesta de trade a jugadores (si w_trade_offer es alto)
        if self.w_trade_offer > 0.5 and needed_material is not None:
            # Ofrecemos algo que tengamos en exceso a cambio de lo que necesitamos
            for material_id in [MaterialConstants.WOOL, MaterialConstants.CLAY,
                                 MaterialConstants.WOOD, MaterialConstants.CEREAL,
                                 MaterialConstants.MINERAL]:
                amount = self.hand.resources.get_from_id(material_id)
                if amount >= 2 and material_id != needed_material:
                    gives = Materials(0, 0, 0, 0, 0)
                    gives.add_from_id(material_id, 1)
                    receives = Materials(0, 0, 0, 0, 0)
                    receives.add_from_id(needed_material, 1)
                    return TradeOffer(gives, receives)

        return None
    
    def on_commerce_response(self, board_instance, commerce_offer):
        """
        ESTA ES LA QUE TE FALTA Y HACE QUE PETE EL PROGRAMA.
        El motor la llama para confirmar qué material quieres recibir.
        """
        needed = self._get_most_needed_material()
        # Si no sabemos qué queremos, pedimos Cereal (0) por defecto para no devolver None
        if needed is None:
            return 0 
        return needed

    def on_trade_offer(self, board_instance, offer=TradeOffer(), player_id=int):
        """
        Evaluamos si aceptar una oferta recibida.
        w_trade_accept controla qué tan generosos somos:
        - alto: aceptamos si recibimos al menos lo mismo que damos
        - bajo: solo aceptamos si recibimos más de lo que damos
        """
        if offer is None:
            return False

        # Contamos materiales que damos y recibimos
        gives_total = (offer.gives.cereal + offer.gives.mineral +
                       offer.gives.clay + offer.gives.wood + offer.gives.wool)
        receives_total = (offer.receives.cereal + offer.receives.mineral +
                          offer.receives.clay + offer.receives.wood + offer.receives.wool)

        # ¿Lo que recibimos nos acerca a nuestro objetivo?
        needed = self._get_most_needed_material()
        receives_needed = offer.gives.get_from_id(needed) if needed is not None else 0

        if self.w_trade_accept > 0.6:
            # Generosos: aceptamos si recibimos algo útil aunque sea igual
            return gives_total <= receives_total or receives_needed > 0
        else:
            # Conservadores: solo aceptamos si recibimos más de lo que damos
            return gives_total < receives_total

    def on_build_phase(self, board_instance):
        """
        Fase de construcción. Elegimos qué construir según los pesos
        w_city, w_town, w_road, w_card, ajustados por disponibilidad de recursos.

        Lógica:
        1. Calculamos qué podemos construir con los recursos actuales
        2. Puntuamos cada opción con su peso * factores del estado del juego
        3. Ejecutamos la opción con mayor puntuación
        """
        self.board = board_instance

        # Modificador dinámico de w_card basado en w_buy_card
        # Si w_buy_card es alto, aumentamos la preferencia por comprar carta
        effective_w_card = self.w_card * (1 + self.w_buy_card)

        opciones = {}  # {nombre: puntuación}

        # --- Ciudad ---
        if self.hand.resources.has_more(BuildConstants.CITY):
            valid = self.board.valid_city_nodes(self.id)
            if valid:
                # Bonus si ya tenemos pueblos que mejorar
                bonus = 1.5 if self.town_count > 0 else 0.5
                opciones['city'] = self.w_city * bonus

        # --- Pueblo ---
        if self.hand.resources.has_more(BuildConstants.TOWN):
            valid = self.board.valid_town_nodes(self.id)
            if valid:
                # Bonus si tenemos buenos nodos disponibles
                best_score = max(self._score_node(n) for n in valid)
                bonus = 1.0 + (best_score / 10.0)  # normalizado
                opciones['town'] = self.w_town * bonus

        # --- Carretera ---
        if self.hand.resources.has_more(BuildConstants.ROAD):
            valid = self.board.valid_road_nodes(self.id)
            if valid:
                # Carretera es menos útil si ya tenemos muchas
                # Penalizamos si el número de carreteras supera mucho al de pueblos
                road_count = sum(1 for node in self.board.nodes
                                 for road in node['roads'] if road['player_id'] == self.id)
                penalty = max(0.3, 1.0 - road_count * 0.05)
                opciones['road'] = self.w_road * penalty

        # --- Carta de desarrollo ---
        if self.hand.resources.has_more(BuildConstants.CARD):
            opciones['card'] = effective_w_card

        if not opciones:
            return None

        # Elegimos la opción con mayor puntuación
        best_action = max(opciones, key=opciones.get)

        # Ejecutamos la acción elegida
        if best_action == 'city':
            valid = self.board.valid_city_nodes(self.id)
            best_node = max(valid, key=lambda n: self._score_node(n))
            self.town_count -= 1
            return {'building': BuildConstants.CITY, 'node_id': best_node}

        elif best_action == 'town':
            valid = self.board.valid_town_nodes(self.id)
            best_node = max(valid, key=lambda n: self._score_node(n))
            self.town_count += 1
            return {'building': BuildConstants.TOWN, 'node_id': best_node}

        elif best_action == 'road':
            valid = self.board.valid_road_nodes(self.id)
            # Elegimos la carretera que lleve al nodo con mejor puntuación
            best_road = max(valid, key=lambda r: self._score_node(r['finishing_node']))
            return {
                'building': BuildConstants.ROAD,
                'node_id': best_road['starting_node'],
                'road_to': best_road['finishing_node']
            }

        elif best_action == 'card':
            return {'building': BuildConstants.CARD}

        return None

    def on_turn_end(self):
        """
        Al final del turno: jugamos carta de punto de victoria si tenemos.
        """
        if not self.development_cards_hand.hand:
            return None

        for i, card in enumerate(self.development_cards_hand.hand):
            if card.type == DevelopmentCardConstants.VICTORY_POINT:
                return self.development_cards_hand.select_card(i)

        return None

    def on_monopoly_card_use(self):
        """
        Carta de monopolio: reclamamos el material que más necesitamos.
        """
        needed = self._get_most_needed_material()
        return needed if needed is not None else MaterialConstants.MINERAL

    def on_road_building_card_use(self):
        """
        Carta de construcción de carreteras: construimos las dos mejores.
        """
        valid = self.board.valid_road_nodes(self.id)
        if not valid:
            return None

        scored = sorted(valid,
                        key=lambda r: self._score_node(r['finishing_node']),
                        reverse=True)

        if len(scored) >= 2:
            return {
                'node_id': scored[0]['starting_node'],
                'road_to': scored[0]['finishing_node'],
                'node_id_2': scored[1]['starting_node'],
                'road_to_2': scored[1]['finishing_node'],
            }
        else:
            return {
                'node_id': scored[0]['starting_node'],
                'road_to': scored[0]['finishing_node'],
                'node_id_2': None,
                'road_to_2': None,
            }

    def on_year_of_plenty_card_use(self):
        """
        Carta de año de la cosecha: tomamos los dos materiales que más necesitamos.
        """
        needed = self._get_needed_materials_sorted()
        m1 = needed[0] if len(needed) > 0 else MaterialConstants.MINERAL
        m2 = needed[1] if len(needed) > 1 else MaterialConstants.CEREAL
        return {'material': m1, 'material_2': m2}

    # =========================================================================
    # FUNCIONES AUXILIARES INTERNAS
    # =========================================================================

    def _get_most_needed_material(self):
        """
        Versión corregida: si no falta nada urgente, devuelve el material que 
        menos tengamos para no romper el motor con un None.
        """
        # ¿Qué nos falta para ciudad? (3 mineral + 2 cereal)
        if self.town_count > 0:
            if self.hand.resources.mineral < 3:
                return MaterialConstants.MINERAL
            if self.hand.resources.cereal < 2:
                return MaterialConstants.CEREAL

        # ¿Qué nos falta para pueblo? (1 madera + 1 arcilla + 1 lana + 1 cereal)
        for mat, needed in [
            (MaterialConstants.WOOD, 1),
            (MaterialConstants.CLAY, 1),
            (MaterialConstants.WOOL, 1),
            (MaterialConstants.CEREAL, 1),
        ]:
            if self.hand.resources.get_from_id(mat) < needed:
                return mat

        # --- AQUÍ ESTABA EL ERROR ---
        # En lugar de return None, devolvemos el material del que tengamos menos cantidad
        # así el motor siempre recibe un entero válido.
        mats = [0, 1, 2, 3, 4]
        return min(mats, key=lambda m: self.hand.resources.get_from_id(m))

    def _get_needed_materials_sorted(self):
        """
        Devuelve lista de materiales ordenados de más necesario a menos.
        Útil para Year of Plenty y Monopolio.
        """
        # Definimos cuánto queremos de cada material para el objetivo ciudad
        targets = {
            MaterialConstants.MINERAL: 3,
            MaterialConstants.CEREAL: 2,
            MaterialConstants.WOOD: 1,
            MaterialConstants.CLAY: 1,
            MaterialConstants.WOOL: 1,
        }
        deficits = {}
        for mat, target in targets.items():
            current = self.hand.resources.get_from_id(mat)
            deficits[mat] = max(0, target - current)

        # Ordenamos por déficit descendente
        return sorted(deficits.keys(), key=lambda m: deficits[m], reverse=True)
