#from zadanie1 import ts

from nolds import sampen
from nolds import corr_dim
from nolds import hurst_rs
import numpy as np

ts = np.random.normal(size=500)

print(ts)

hurst_exponent = hurst_rs(ts)
print("Wykładnik Hursta: ", hurst_exponent)


fractal_dimension = corr_dim(ts, emb_dim=2)
print("Wymiar fraktalny: ", fractal_dimension)


entropy = sampen(ts)
print("Entropia: ", entropy)


"""
Entropia - mierzy stopień nieuporządkowania i przypadkowości w szeregu czasowym
np finanse = zmiennośc rynku
-ocena skomplikowanego systemu
-pomga w wykrywaniu ukrytych wzorców

Wymiar fraktalny - określa stopień skomplikowania geometrycznego szeregu czasowego, 
tzn. jak bardzo jego krzywizny i zakamarki są nieregularne
finanse = niestabilnośc rynku
-
Wykładnik Hursta - mierzy stopień długoterminowej zależności między wartościami szeregu czasowego. 
Może służyć do oceny "płynności" szeregu czasowego.
H < 0,5 oznacza anty-persystencję, wartości szeregu czasowego są bardziej losowe i niestabilne.
H = 0,5 oznacza brak persystencji, wartości szeregu czasowego są losowe i niezależne od siebie.
H > 0,5 oznacza persystencję, wartości szeregu czasowego są bardziej stabilne i skłonne do utrzymywania się na podobnym poziomie przez dłuższy czas.

"""