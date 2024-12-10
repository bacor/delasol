# Geneval Psalter

This corpus contains transcriptions of the psalms from the 1562 Genevan Psalter by Marot and de Bèze:

Clément Marot and Théodore de Bèze, *Les Psaumes en vers français avec leurs mélodies*, (Geneva: Michel Blanchier, 1562; facsimile ed. by Pierre Pidoux, (Geneva: Librairie Droz, 2nd ed. 1986)

These are meant as diplomatic editions, not correcting mistakes (although corrections may be indicated). Transcribed are note music, solmization syllables, the original lyrics and lyrics in modern spelling. The modern spelling and hyphenation is taken from the 1895 edition of Goudimels psalm settings.

## Transcription guidelines

### Music
- **Clef.** Transcribe music in the original clef
- **Time signature.** Typically transcribed in 4/2 meter
- **Barlines.** Use barlines in the transcription for easy reference.
- **Last note** The duration of the last note is shown as a double whole note, even when it is a longa in the source, and irrespective of whether it fits in the bar
- **No layout.** Do not edit the layout to keep the musicxml files clean.

### Metadata
- **Title & lyricist**. Psalms have titles of the form:  
  PSEAUME XXVI  
	TH. DE BE.  
  This is transcribed as “Psalm 26” (title) and “Théodore de Bèze” (lyricist). Similarly, “CL. MA.” results in lyricist “Clement Marot”
- **Subtitle.** The first phrase in modern French spelling
- **Composer** is not transcribed.
- **Page numbers** shown at the top left of the score correspond to the entire psalm, including all verses.
- **Editorial comments.** An optional text frame at the end can discuss editorial choices.
- **Project Properties.** Add title, subtitle, lyricist, copyright as metadata to the project properties. Standard copyright message `Copyright © Bas Cornelissen 2024; CC BY-SA 4.0`. Insert the work id (e.g. `MdB026`) in the field 'Work number' and add a new property 'Editor' with the editor (e.g. `Bas Cornelissen`).

### Lyrics
- **Lyrics line 1**: The original lyrics, without alterations. The original lyrics are often printed as whole words, then the words will be split following modern hyphenation, but hyphens are omitted. Hyphens in the original are included as en-dashes (–) at the previous syllable.
- **Lyrics line 2**: modernized spelling including hyphenation, following the 1895 edition of Goudimels psalm settings. This line is set in italics.
- **Lyrics line 3**: solmization syllables (see below)

### Solmization syllables

- Valid syllables: `ut, re, mi, fa, sol, la`
- `?` indicates missing solmization syllable
- `?mi` probably a mi, but there is some uncertainty (e.g. not readable)
- `!re[fa]` indicates that there is a re in the source, but this is likely a mistake
- `!?re[?fa]` indicates that there is a mistake in the source, probably a re, which should probably be a fa

A regular expression matching this scheme:

```re
\!?(\??)(ut|re|mi|fa|sol|la)?(\[(\??)(ut|re|mi|fa|sol|la)\])?
```

## Goudimel index

Index to the psalms in the 1895 Goudimel edition.
The final column indicates whether these guidelines are respected.

| Psalm (MdB) | Goudimel num.   | Volume | Page | Title                                         | Guidelines |
| ----------- | --------------- | ------ | ---- | --------------------------------------------- |---|
| 1           | I               | 1      | 1    | Qui au conseil des malins n'a esté            | no |
| 2           | XXXIX           | 1      | 102  | Pourquoi font bruit et s'assemblent les gens? | no |
| 3           | XVIII           | 1      | 45   | O Seigneur, que de gens                       | no |
| 4           | CXLIV           | 3      | 115  | Quand je t'invoque, helas, escoute            | no |
| 5           | XL              | 1      | 105  | Aux paroles que je eux dire                   | no |
| 6           | CXLV            | 3      | 119  | Ne veuilles pas, ô Sire                       | no |
| 7           | LXXXVIII        | 2      | 90   | Mon Dieu, j'ai en toi esperance               | no |
| 8           | XLI             | 1      | 107  | O nostre Dieu et Seigneur amiable             | no |
| 9           | XLIII           | 1      | 112  | De tout mon coeur t'exalterai                 | no |
| 10          | XLII            | 1      | 109  | D'ou vient cela, Seigneur je te suppli'       | no |
| 11          | XLIV            | 1      | 114  | Veu que du tout en Dieu mon coeur s'appuie    | no |
| 12          | XLV             | 1      | 117  | Donne secours, Seigneur, il en est heure      | no |
| 13          | XLVI            | 1      | 119  | Jusques à quand as establi                    | no |
| 14          | XLVII           | 1      | 121  | Le fol malin en soun coeur dit et croit       | no |
| 15          | CXXVIII         | 3      | 75   | Qui est-ce qui conversera                     | no |
| 16          | XCVII           | 2      | 114  | Sois moi, Seigneur, ma garde et mon appui     | no |
| 17          | XCVIII          | 2      | 117  | Seigneur, enten à mon bon droict              | no |
| 18          | CXLVI           | 3      | 121  | Je t'aimerai en toute obeissance              | no |
| 19          | CXVIII          | 3      | 44   | Les cieux en chacun lieu                      | no |
| 20          | XVIII           | 1      | 123  | Le Seigneur ta prière entende                 | no |
| 21          | VI              | 1      | 16   | Seigneur, le Roi s'esjouira                   | no |
| 22          | CXLIX           | 3      | 133  | Mon Dieu, mon Dieu, pourquoi m'as tu laissé   | no |
| 23          | LXXXIX          | 2      | 93   | Mon Dieu me paist sous sa puissance haute     | no |
| 24          | XLIX            | 1      | 125  | La terre au Seigneur appartient               | no |
| 25          | XIX             | 1      | 48   | A toi, mon Dieu, mon coeur monte              | no |
| 37          | LII             | 2      | 4    | Ne sois fasché si durant ceste vie            | no |
| 26          | XCIX            | 2      | 120  | Seigneur, garde mon droit                     | yes |
| 27          | CXIX            | 3      | 48   | Le Seigneur est la clairté qui m'adresse      | yes |
| 28          | XC              | 2      | 96   | O Dieu, qui es ma forteresse                  | yes |