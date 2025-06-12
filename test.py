import music21 as m21
from delasol import solmize

import matplotlib.pyplot as plt

score = m21.converter.parse('tests/scores/MdB001.musicxml')
solfa = solmize(score, style="tinctoris_15c")
solfa.annotate()
