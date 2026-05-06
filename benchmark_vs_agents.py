# benchmark_vs_agents.py
import os, importlib, concurrent.futures
from Managers.GameDirector import GameDirector

MI_AGENTE = "Agents.DenysAgent.DenysAgent"
RIVALES = [
    "Agents.AdrianHerasAgent.AdrianHerasAgent",
    "Agents.SigmaAgent.SigmaAgent",
    "Agents.TristanAgent.TristanAgent",
    "Agents.CarlesZaidaAgent.CarlesZaidaAgent",
    "Agents.EdoAgent.EdoAgent",
    "Agents.PabloAleixAlexAgent.PabloAleixAlexAgent",
    "Agents.CrabisaAgent.CrabisaAgent",
]
N = 5  # Número de partidas por combinación

def cargar(r): mod,cls=r.rsplit(".",1); return getattr(importlib.import_module(mod),cls)

def jugar(args):
    pos, mi, r1, r2, r3 = args
    class A(mi):
        def __init__(self,id): super().__init__(id)
    agents=[r1,r2,r3]; agents.insert(pos,A)
    try:
        gd=GameDirector(agents=agents,max_rounds=200,store_trace=False)
        t=gd.game_start(print_outcome=False)
        lr=max(t["game"].keys(),key=lambda r:int(r.split("_")[-1]))
        lt=max(t["game"][lr].keys(),key=lambda x:int(x.split("_")[-1].lstrip("P")))
        vp=t["game"][lr][lt]["end_turn"]["victory_points"]
        k=f"J{pos}"; w=max(vp,key=lambda x:int(vp[x]))
        return 1 if w==k else 0
    except: return 0

if __name__=='__main__':
    mi=cargar(MI_AGENTE)
    rs=[cargar(r) for r in RIVALES]
    wins=tot=0
    workers=max(1,int(os.cpu_count()*0.4))
    args=[(pos,mi,rs[i],rs[j],rs[k])
          for i in range(len(rs)) for j in range(i+1,len(rs))
          for k in range(j+1,len(rs)) for pos in range(4)
          for _ in range(N)]
    print(f"Total partidas: {len(args)}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(jugar,args): wins+=r; tot+=1
    print(f"Victorias: {wins}/{tot} = {wins/tot:.2%}")
    print(f"Objetivo: >25%")