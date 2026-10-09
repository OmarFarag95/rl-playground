from .diver import Diver
from .dog import Dog
from .pizza import Pizza
from .sandwich import Sandwich
from .walker import Walker

GAMES = {g.key: g for g in (Diver, Dog, Pizza, Sandwich, Walker)}
