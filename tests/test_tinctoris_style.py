import music21 as m21
from delasol import solmize, MutationFormatter
from delasol.corpus import Corpus

import matplotlib.pyplot as plt

score = m21.converter.parse('tests/scores/MdB001.musicxml')
solfa = solmize(score, style="tinctoris_15c")
solfa.annotate(format="mutation")
# solfa.stream.show()
# solfa.rollout.draw()
# plt.show()
