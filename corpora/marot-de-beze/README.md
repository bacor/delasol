# Geneval Psalter

This corpus contains transcriptions of the psalms from the 1562 Genevan Psalter by Marot and de Bèze:

Clément Marot and Théodore de Bèze, *Les Psaumes en vers français avec leurs mélodies*, (Geneva: Michel Blanchier, 1562; facsimile ed. by Pierre Pidoux, (Geneva: Librairie Droz, 2nd ed. 1986)

These are meant as more or less diplomatic editions, in principle not correcting mistakes. Whenever corrections have been made, this is indicated in the editorial comments. Transcriptions include the music, solmization syllables, the original lyrics and lyrics in modern spelling. The modern spelling and hyphenation is taken from the 1895 edition of Goudimels psalm settings.

## Transcription guidelines

### Music
- **Note shapes.** Modernize note shapes (round instead of diamond).
- **Clef.** Transcribe music in the original clef.
- **Time signature.** Transcribe psalms in the original time signature: usually that is in cut time or in a 4/2 meter.
- **Barlines.** Use ordinary barlines in the transcription for easy reference (and not, say, Mensuralstriche). This might occassionally introduce ties when whole notes cross barlines. The final barline is transcribed as a double barline.
- **Ties.** All ties are editorial additions to transcribe notes across barlines, unless noted otherwise in the editorial comments.
- **Last note** The duration of the last note is shown as a double whole note, even when it is a longa in the source, and irrespective of whether it fits in the bar
- **Breaks & custos.** System breaks are indicated by placing a single dash caesura after the last note of a system; page breaks by a double dash caesura. At those places, a custos _corresponding to the next note_ appears in the source, unless explicitly noted otherwise in the editorial comments. This means custos not transcribed explicitly. Example of a editorial comment: `m. 10. The custos is inconsistent: it indicates an A instead of an E`.
- **No layout.** Do not edit the layout to keep the musicxml files clean. In particular, system and page breaks are not transcribed as such (but using caesura, see above).

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
- **Lyrics line 1**: The original lyrics, without alterations. 
  - In the source, the very first letter of a psalm is set as a large capital before the score. Such capitals are bold and underlined in the transcription.
  - If syllables in the original are clearly separated, that hyphenation will be used, _without adding hyphens_. If words are printed as whole words and the hyphenation is unclear, modern hyphenation is used and a tilde `~` is added instead of a hyphen to indicate the editorial change. Finally, hyphens in the original are transcribed using en-dashes (`—`); these mostly  occur in words across system breaks or in for example `doy-ie` (_doi je_)
  - When an n is omitted and written as a tilde-like symbol above the previous vowel, transcribe it like so: `d'hõneurs` (for the modern _d'honneurs_), `dõc` (_donc_), etc.
  - Transcribe a `u` when used instead of `v`, as in `deliurance` (_delivrance_ )
  - Transcribe `œ` instead of `oe` if the ligature appears in the source.
  - Ignore discretionary ligatures (e.g. on ct)
  - Do extend lyrics with underscores for tied notes.
- **Lyrics line 2**: modernized spelling including hyphenation, following the 1895 edition of Goudimels psalm settings. 
  - This line is set in italics.
  - Copy the interpunction from the 1895 edition, even if it differs from the interpunction in the original. 
  - Do extend lyrics with underscores for tied notes (or, as usual, normal dashes if they occur within a word).
- **Lyrics line 3**: solmization syllables (see below). 
  - Do extend syllables with underscores for tied notes.

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

| Psalm (MdB) | Goudimel num.   | Volume | Page | Title                                         |
| ----------- | --------------- | ------ | ---- | --------------------------------------------- |
| 1           | I               | 1      | 1    | Qui au conseil des malins n'a esté            |
| 2           | XXXIX           | 1      | 102  | Pourquoi font bruit et s'assemblent les gens? |
| 3           | XVIII           | 1      | 45   | O Seigneur, que de gens                       |
| 4           | CXLIV           | 3      | 115  | Quand je t'invoque, helas, escoute            |
| 5           | XL              | 1      | 105  | Aux paroles que je eux dire                   |
| 6           | CXLV            | 3      | 119  | Ne veuilles pas, ô Sire                       |
| 7           | LXXXVIII        | 2      | 90   | Mon Dieu, j'ai en toi esperance               |
| 8           | XLI             | 1      | 107  | O nostre Dieu et Seigneur amiable             |
| 9           | XLIII           | 1      | 112  | De tout mon coeur t'exalterai                 |
| 10          | XLII            | 1      | 109  | D'ou vient cela, Seigneur je te suppli'       |
| 11          | XLIV            | 1      | 114  | Veu que du tout en Dieu mon coeur s'appuie    |
| 12          | XLV             | 1      | 117  | Donne secours, Seigneur, il en est heure      |
| 13          | XLVI            | 1      | 119  | Jusques à quand as establi                    |
| 14          | XLVII           | 1      | 121  | Le fol malin en soun coeur dit et croit       |
| 15          | CXXVIII         | 3      | 75   | Qui est-ce qui conversera                     |
| 16          | XCVII           | 2      | 114  | Sois moi, Seigneur, ma garde et mon appui     |
| 17          | XCVIII          | 2      | 117  | Seigneur, enten à mon bon droict              |
| 18          | CXLVI           | 3      | 121  | Je t'aimerai en toute obeissance              |
| 19          | CXVIII          | 3      | 44   | Les cieux en chacun lieu                      |
| 20          | XVIII           | 1      | 123  | Le Seigneur ta prière entende                 |
| 21          | VI              | 1      | 16   | Seigneur, le Roi s'esjouira                   |
| 22          | CXLIX           | 3      | 133  | Mon Dieu, mon Dieu, pourquoi m'as tu laissé   |
| 23          | LXXXIX          | 2      | 93   | Mon Dieu me paist sous sa puissance haute     |
| 24          | XLIX            | 1      | 125  | La terre au Seigneur appartient               |
| 25          | XIX             | 1      | 48   | A toi, mon Dieu, mon coeur monte              |
| 26          | XCIX            | 2      | 120  | Seigneur, garde mon droit                     |
| 27          | CXIX            | 3      | 48   | Le Seigneur est la clairté qui m'adresse      |
| 28          | XC              | 2      | 96   | O Dieu, qui es ma forteresse                  |
| 29          | XVII            | 1      | 42   | Vous tous, Princes et Seigneurs               |
| 30          | CXXIX           | 3      | 77   | Seigneur, puis que m'as retiré.               |
| 31          | C               | 2      | 122  | J'ai mis en toi mon esperance                 |
| 32          | II              | 1      | 4    | O bien heureux celui, dont les commises       |
| 33          | L               | 1      | 127  | Réveillez-vous chacun fidèle                  |
| 34          | LI              | 2      | 2    | Jamais ne cesserai                            |
| 35          | XX              | 1      | 51   | Deba contre mes debateurs                     | 
| 36          | IV              | 1      | 10   | Du malin le meschant vouloir                  |
| 37          | LII             | 2      | 4    | Ne sois fasché si durant ceste vie            |
| 38          | CXLVIII         | 3      | 131  | Las, en ta fureur aigue                       |

## Progress

The second column indicates whether the transcription guidelines in this document are respected.


| Psalm (MdB) | New guidelines? | Proofread (Bas) |
| ----------- | --------------- | --------------- |
| 1           | ❌              |                 |
| 2           | ❌              |                 |
| 3           | ❌              |                 |
| 4           | ❌              |                 |
| 5           | ❌              |                 |
| 6           | ❌              |                 |
| 7           | ❌              |                 |
| 8           | ❌              |                 |
| 9           | ❌              |                 |
| 10          | ❌              |                 |
| 11          | ❌              |                 |
| 12          | ❌              |                 |
| 13          | ❌              |                 |
| 14          | ❌              |                 |
| 15          | ❌              |                 |
| 16          | ❌              |                 |
| 17          | ❌              |                 |
| 18          | ❌              |                 |
| 19          | ❌              |                 |
| 20          | ❌              |                 |
| 21          | ❌              |                 |
| 22          | ❌              |                 |
| 23          | ❌              |                 |
| 24          | ❌              |                 |
| 25          | ❌              |                 |
| 37          | ❌              |                 |
| 26          | ✅              |                 |
| 27          | ✅              |                 |
| 28          | ✅              |                 |
| 29          | ✅              |                 |
| 30          | ✅              |                 |
| 31          | ✅              |                 |
| 32          | ✅              |                 |
| 33          | ✅              |                 |
| 34          | ✅              |                 |
| 35          | ✅              |                 |
| 36          | ✅              |                 |
| 37          | ✅              |                 |
| 38          | ✅              |                 |