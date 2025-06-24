from pathlib import Path
import music21 as m21
from delasol import solmize, MutationFormatter
from delasol.corpus import Collection
from delasol.evaluator import Evaluator

import matplotlib.pyplot as plt

# To do: make it a real test.

# score = m21.converter.parse('tests/scores/MdB001.musicxml')
# solfa = solmize(score, style="tinctoris_15c")
# solfa.annotate(format="mutation")
# solfa.stream.show()
# solfa.rollout.draw()
# plt.show()

TINCTORIS_CORPUS = Path("./../delasol-corpus/drafts/tinctoris-c1475-all-syllables/")

def evaluate_tinctoris(tinctoris_collection_path=TINCTORIS_CORPUS):
    tinctoris_collection = Collection(directory=str(tinctoris_collection_path))
    # Based on tests/test_corpus, lines 176-179
    ids = tinctoris_collection.ids
    df, log = tinctoris_collection.evaluate(style="tinctoris_15c",\
        # ids=ids,\
        target_lyric_number=2,\
        # write_output=True,\
        # output_dir=str(Path('.'))\
    )
    return df, log

df, log = evaluate_tinctoris(); print(df)
exit()

piece = 'JT006.musicxml'
score = m21.converter.parse(TINCTORIS_CORPUS / 'musicxml' / piece)
# score = m21.converter.parse('tests/scores/MdB001.musicxml')

# solfa = solmize(score, style="tinctoris_15c")
try:
    solfa = solmize(score, style="tinctoris_15c")
    print(solfa.evaluate(target_lyric_number=1))
except:
    score.show()
# solfa.annotate(format="syllable")
# solfa.annotate(format="mutation")
# solfa.gamut.draw()
# solfa.rollout.draw()
# solfa.stream.show()
# plt.show()
