🌞 Delasol — automatic hexachordal solmization
===================================

**The Delasol project aims to develop methods for automatic hexachordal solmization. Currenty it supports sixteenth-century continental and English solmization styles.**

Solmization is modeled as a walk through a so-called _gamut graph_. A gamut graph describes possible melodic movements through the gamut. In the sixteenth-century, 'hard' keys without flats in the key signature are solmized using only hard and natural hexachords, whereas 'soft' keys use soft and natural hexachords. These result in two different gamut graphs. The 'continental' solmization style mutates up on the _re_ of the next hexachord, and down on _la_ of the next hexachord. This is different from the solmization style used in England around the same time. Such stylistic differences are represented by different gamut graphs. 

To produce a solmization for a given melody in a given style,  _Delasol_ looks for the cheapest path through the gamut graph that traverses the melodies' pitches. Instead of comparing all paths globally, the melody is divided into _segments_ of which the start and endpoint have only one possible solmization. The (global) cheapest path is then found by combining the best (local) paths of all segments.

Evaluating the model can be done by comparing it to reference solmizations. 
To that end we have transcribed solmizations from several sixteenth-century psalters, such as the Geneva Psalter (1562) for continental style, and the Whole Book of Psalms (1590) for English style.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   docs
   api
