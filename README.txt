Usage:

    python findtrc.py

Requires Python 3.6+, NLTK (`pip install nltk`), and a copy of the [Ubuntu Chat
Corpus](https://daviduthus.org/UCC/).

This script detects instances of text repair from the Ubuntu Chat Corpus(1). 
Example of text repair:

    [18:59] <deebee396> ... this takes quite along time ...
    [18:59] <deebee396> *a long *boot

This is a repair event for `along -> a long` and `time -> boot time`.

This script implements an algorithm to detect repair events, and annotates them
with the markup schema for the Text Repair Corpus.

Repairs are detected by a combination of heuristics, including message- and
word-level edit distance, turn distance between repair and a candidate target,
use of markers such as *. Each heuristic contributes a weighted value to the
confidence score for the repair. Each linked target and repair are filtered for
a minimum confidence score before inclusion in the Text Repair Corpus.

As of writing, the script takes about 3 ~ 4 seconds to evaluate a medium-size
(~200 kb) file in the UCC. Each file may yield multiple repair events.

(1): Uthus, D., & Aha, D. (2013). The Ubuntu Chat Corpus for Multiparticipant
Chat Analysis. In 2013 AAAI Spring Symposium Series.
https://www.aaai.org/ocs/index.php/SSS/SSS13/paper/view/5706
