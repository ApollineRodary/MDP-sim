import ctypes

mdp = ctypes.CDLL("lib/mdp.dll")
mdp.CreateMDP(3, None, None, None)
print("Ok done")