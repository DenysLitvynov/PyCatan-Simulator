"""
genetic_algorithm.py
====================
Entrena MiAgente usando un algoritmo genético contra TODOS los agentes disponibles.

Estrategia de entrenamiento:
- Cada individuo se evalúa jugando contra combinaciones variadas de rivales
- Los rivales se eligen aleatoriamente de un pool que incluye todos los agentes
- Esto obliga al agente a aprender estrategias robustas que funcionen contra cualquier rival
- RandomAgent se incluye pero con menos peso que los agentes inteligentes

Uso:
    python genetic_algorithm.py
"""

import random
import json
import time
import os
import importlib
import concurrent.futures

from Managers.GameDirector import GameDirector
from Agents.MiAgente import MiAgente, CHROMOSOME_SIZE, random_chromosome

# =============================================================================
# POOL DE RIVALES — todos los agentes disponibles
# Añade o elimina según los que tengas en tu carpeta Agents/
# =============================================================================

RIVAL_AGENTS_PATHS = [
    "Agents.RandomAgent.RandomAgent",
    "Agents.AdrianHerasAgent.AdrianHerasAgent",
    "Agents.SigmaAgent.SigmaAgent",
    "Agents.TristanAgent.TristanAgent",
    "Agents.CarlesZaidaAgent.CarlesZaidaAgent",
    "Agents.EdoAgent.EdoAgent",
    "Agents.PabloAleixAlexAgent.PabloAleixAlexAgent",
    "Agents.CrabisaAgent.CrabisaAgent",
    "Agents.AlexPastorAgent.AlexPastorAgent",
    "Agents.EdoAgent.EdoAgent"   
]

# =============================================================================
# HIPERPARÁMETROS
# =============================================================================

POPULATION_SIZE   = 30    # individuos por generación
N_GENERATIONS     = 50    # generaciones totales
GAMES_PER_EVAL    = 16    # partidas por individuo para calcular fitness
                           # con más rivales necesitamos más partidas para que sea representativo
ELITE_SIZE        = 4     # mejores que pasan directos a la siguiente generación
CROSSOVER_PROB    = 0.8   # probabilidad de cruce
MUTATION_PROB     = 0.15  # probabilidad de mutar cada gen
MUTATION_STRENGTH = 0.2   # magnitud de la mutación gaussiana
TOURNAMENT_SIZE   = 3     # tamaño del torneo de selección
MAX_ROUNDS        = 200   # rondas máximas por partida
WORKER_RATIO      = 0.75  # porcentaje de cores a usar

OUTPUT_FILE = "best_chromosome.json"

# =============================================================================
# CARGA DE AGENTES RIVALES
# =============================================================================

def cargar_agente(ruta_clase):
    """Carga dinámicamente una clase de agente desde su ruta de módulo."""
    modulo, clase = ruta_clase.rsplit(".", 1)
    mod = importlib.import_module(modulo)
    return getattr(mod, clase)


def cargar_todos_los_rivales():
    """
    Carga todos los agentes del pool. Si alguno falla (fichero no encontrado,
    error de importación), lo omite y avisa.
    Devuelve lista de clases de agente.
    """
    rivales = []
    for ruta in RIVAL_AGENTS_PATHS:
        try:
            agente = cargar_agente(ruta)
            rivales.append(agente)
            print(f"  [OK] Cargado rival: {agente.__name__}")
        except Exception as e:
            print(f"  [--] No se pudo cargar {ruta}: {e}")
    return rivales


# =============================================================================
# EVALUACIÓN DEL FITNESS
# =============================================================================

def evaluate_individual(args):
    """
    Evalúa un cromosoma lanzando GAMES_PER_EVAL partidas.

    Estrategia de selección de rivales:
    - En cada partida se eligen 3 rivales ALEATORIAMENTE del pool
    - RandomAgent tiene menos probabilidad de ser elegido (peso 0.3)
      que los agentes inteligentes (peso 1.0 cada uno)
    - Esto asegura que el agente aprende contra variedad de oponentes

    Fitness compuesto:
        0.60 * win_rate         -> ganar es lo mas importante
        0.30 * avg_vp / 10.0   -> tener muchos puntos aunque no ganes
        0.10 * rank_score       -> no terminar ultimo

    Devuelve float. Mayor = mejor.
    """
    cromosoma, rival_names, n_games = args

    # Recargamos los rivales dentro del worker (necesario para multiprocessing)
    rival_classes = []
    for ruta in rival_names:
        try:
            rival_classes.append(cargar_agente(ruta))
        except Exception:
            pass

    if not rival_classes:
        return 0.0

    total_wins   = 0
    total_vp     = 0
    total_rank   = 0
    games_played = 0

    # Pesos para selección aleatoria: Random vale menos que agentes inteligentes
    weights = [0.3 if cls.__name__ == "RandomAgent" else 1.0 for cls in rival_classes]

    # Jugamos en las 4 posiciones repartiendo las partidas
    games_per_position = max(1, n_games // 4)

    for pos in range(4):
        for _ in range(games_per_position):
            tres_rivales = random.choices(rival_classes, weights=weights, k=3)
            try:
                win, vp, rank = _play_one_game(cromosoma, pos, tres_rivales)
                total_wins  += win
                total_vp    += vp
                total_rank  += rank
                games_played += 1
            except Exception:
                total_rank  += 4
                games_played += 1

    if games_played == 0:
        return 0.0

    win_rate = total_wins / games_played
    avg_vp   = total_vp   / games_played
    avg_rank = total_rank / games_played

    fitness = (
        0.60 * win_rate +
        0.30 * (avg_vp / 10.0) +
        0.10 * (1.0 - (avg_rank - 1.0) / 3.0)
    )
    return fitness


def _play_one_game(cromosoma, position, rivals):
    """
    Juega una sola partida.
    position: posicion de nuestro agente (0-3)
    rivals: lista de 3 clases de agente rivales
    Devuelve (victoria: 0/1, puntos: int, puesto: 1-4)
    """
    class AgentConCromosoma(MiAgente):
        def __init__(self, agent_id):
            super().__init__(agent_id, cromosoma=cromosoma)

    agents = list(rivals)
    agents.insert(position, AgentConCromosoma)

    gd = GameDirector(agents=agents, max_rounds=MAX_ROUNDS, store_trace=False)
    game_trace = gd.game_start(print_outcome=False)

    last_round = max(game_trace["game"].keys(),
                     key=lambda r: int(r.split("_")[-1]))
    last_turn  = max(game_trace["game"][last_round].keys(),
                     key=lambda t: int(t.split("_")[-1].lstrip("P")))
    vp = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]

    agent_key = f"J{position}"
    my_vp     = int(vp[agent_key])
    winner    = max(vp, key=lambda k: int(vp[k]))
    victory   = 1 if winner == agent_key else 0

    sorted_players = sorted(vp.items(), key=lambda x: int(x[1]), reverse=True)
    rank = next(i + 1 for i, (k, _) in enumerate(sorted_players) if k == agent_key)

    return victory, my_vp, rank


# =============================================================================
# OPERADORES GENÉTICOS
# =============================================================================

def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    """
    Selección por torneo: escoge k candidatos al azar y devuelve el mejor.
    Más robusto que la ruleta porque no depende de la escala absoluta del fitness.
    """
    idx  = random.sample(range(len(population)), k)
    best = max(idx, key=lambda i: fitnesses[i])
    return population[best][:]


def crossover_one_point(pa, pb):
    """
    Cruce de un punto: primera mitad de A + segunda mitad de B.
    """
    if random.random() > CROSSOVER_PROB:
        return pa[:]
    point = random.randint(1, CHROMOSOME_SIZE - 1)
    return pa[:point] + pb[point:]


def mutate(cromosoma):
    """
    Mutación gaussiana: con probabilidad MUTATION_PROB, cada gen recibe
    un pequeño ruido gaussiano. El resultado se recorta a [0, 1].
    """
    mutated = cromosoma[:]
    for i in range(len(mutated)):
        if random.random() < MUTATION_PROB:
            noise = random.gauss(0, MUTATION_STRENGTH)
            mutated[i] = max(0.0, min(1.0, mutated[i] + noise))
    return mutated


# =============================================================================
# GUARDADO PARCIAL
# =============================================================================

def _save_result(chromosome, fitness, history):
    """
    Guarda el mejor resultado hasta ahora en JSON.
    Se llama tras cada mejora para no perder progreso si el proceso se interrumpe.
    """
    output = {
        'best_chromosome': chromosome,
        'best_fitness':    fitness,
        'hyperparameters': {
            'population_size':   POPULATION_SIZE,
            'n_generations':     N_GENERATIONS,
            'games_per_eval':    GAMES_PER_EVAL,
            'elite_size':        ELITE_SIZE,
            'crossover_prob':    CROSSOVER_PROB,
            'mutation_prob':     MUTATION_PROB,
            'mutation_strength': MUTATION_STRENGTH,
            'tournament_size':   TOURNAMENT_SIZE,
            'rival_agents':      RIVAL_AGENTS_PATHS,
        },
        'history': history,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)


# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================

def run_genetic_algorithm():
    n_workers = max(1, int(os.cpu_count() * WORKER_RATIO))

    print(f"\n{'='*62}")
    print(f"  ENTRENAMIENTO GENETICO — MiAgente vs TODOS los rivales")
    print(f"{'='*62}")

    print("\nVerificando rivales disponibles...")
    rival_classes = cargar_todos_los_rivales()

    if not rival_classes:
        print("ERROR: No se pudo cargar ningun rival. Revisa RIVAL_AGENTS_PATHS.")
        return None, []

    print(f"\n  Rivales activos:  {len(rival_classes)}")
    print(f"  Poblacion:        {POPULATION_SIZE} individuos")
    print(f"  Generaciones:     {N_GENERATIONS}")
    print(f"  Partidas/eval:    {GAMES_PER_EVAL}")
    print(f"  Workers:          {n_workers}")
    total = POPULATION_SIZE * GAMES_PER_EVAL * N_GENERATIONS
    print(f"  Partidas totales: ~{total:,}")
    print(f"{'='*62}\n")

    # Pasamos los nombres (strings) a los workers, no las clases,
    # porque las clases no siempre son serializables por multiprocessing
    rival_names = RIVAL_AGENTS_PATHS[:len(rival_classes)]

    population = [random_chromosome() for _ in range(POPULATION_SIZE)]

    history              = []
    best_ever_fitness    = -1.0
    best_ever_chromosome = None

    start_time = time.time()

    for generation in range(N_GENERATIONS):
        gen_start = time.time()

        # Preparamos args para cada worker: (cromosoma, lista_de_nombres, n_games)
        eval_args = [(crom, rival_names, GAMES_PER_EVAL) for crom in population]

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            fitnesses = list(executor.map(evaluate_individual, eval_args))

        # Estadísticas de la generación
        best_idx      = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_fitness  = fitnesses[best_idx]
        avg_fitness   = sum(fitnesses) / len(fitnesses)
        worst_fitness = min(fitnesses)
        best_chrom    = population[best_idx]

        # Guardado parcial si mejoramos el record
        if best_fitness > best_ever_fitness:
            best_ever_fitness    = best_fitness
            best_ever_chromosome = best_chrom[:]
            _save_result(best_ever_chromosome, best_ever_fitness, history)
            marker = " <-- NUEVO RECORD"
        else:
            marker = ""

        history.append({
            'generation':    generation + 1,
            'best_fitness':  round(best_fitness,  4),
            'avg_fitness':   round(avg_fitness,   4),
            'worst_fitness': round(worst_fitness, 4),
        })

        elapsed = time.time() - gen_start
        print(f"Gen {generation+1:3d}/{N_GENERATIONS} | "
              f"Best: {best_fitness:.4f} | "
              f"Avg: {avg_fitness:.4f} | "
              f"Worst: {worst_fitness:.4f} | "
              f"{elapsed:.1f}s{marker}")

        # Elitismo: los mejores pasan directos
        sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        elite = [population[i][:] for i in sorted_idx[:ELITE_SIZE]]

        # Nueva generación por selección + cruce + mutación
        new_population = elite[:]
        while len(new_population) < POPULATION_SIZE:
            pa    = tournament_selection(population, fitnesses)
            pb    = tournament_selection(population, fitnesses)
            child = crossover_one_point(pa, pb)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Resumen final
    total_time = time.time() - start_time
    h, r = divmod(total_time, 3600)
    m, s = divmod(r, 60)

    print(f"\n{'='*62}")
    print(f"  COMPLETADO en {int(h)}h {int(m)}m {int(s)}s")
    print(f"  Mejor fitness global: {best_ever_fitness:.4f}")
    print(f"  Mejor cromosoma:")
    for i, g in enumerate(best_ever_chromosome):
        print(f"    [{i:2d}] {g:.3f}")
    print(f"{'='*62}\n")

    _save_result(best_ever_chromosome, best_ever_fitness, history)
    print(f"Resultados guardados en: {OUTPUT_FILE}")
    return best_ever_chromosome, history


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == '__main__':
    best_chromosome, history = run_genetic_algorithm()

    if best_chromosome is not None:
        print("\nEvolucion del fitness por generacion:")
        print(f"{'Gen':>4} | {'Best':>8} | {'Avg':>8} | {'Worst':>8}")
        print("-" * 40)
        for h in history:
            print(f"{h['generation']:>4} | "
                  f"{h['best_fitness']:>8.4f} | "
                  f"{h['avg_fitness']:>8.4f} | "
                  f"{h['worst_fitness']:>8.4f}")