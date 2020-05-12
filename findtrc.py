# -*- coding: utf-8 -*-

"""
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

"""

from glob import glob
from itertools import product
import os
import re
from typing import Optional

from nltk.tokenize.casual import TweetTokenizer


UCC_GLOB = './ucc/**'  # ** matches any file/dir

OUTDIR = 'trc'

# Number of messages to use as context for searching for repairs
REPAIR_WINDOW_SIZE = 10

MARKERS = dict(
    peripheral={
        r'(?<!\*)(\*)(?!\*)',    # * with no preceding or following *
        r'(?<!不)(错了)',  # 错了 but not 不错了
    },
    infix={
        '=',
        'not', # en
        'non', # es
    },
    sed=('s/', '/', '/'),
    clitic={
        r'(?<!\*)(\*)(?!\*)'
    }
)




# Bind globals to local (used in levenshtein distance)
_abs, _len, _min, _max = abs, len, min, max
_symdif = set.symmetric_difference
_lcopy = list.copy

# Shared tokenizer object
tokenize = TweetTokenizer().tokenize


MSG_TEMPLATE = """
  <message id="{msgid}">
    <author>{author}</author>
    <timestamp>{timestamp}</timestamp>
    <content>
      {content}
    </content>
  </message>
"""

DOC_HEADER = """\
<document id="{docid}" annotation="{annotation}"
          language="{language}" date="{date}"
          source="{source}"
          linestart="{linestart}" lineend="{lineend}">
"""

DOC_FOOTER = """
</document>
"""



def markers_matchers():
    head, mid, tail = [], [], []

    for patt in MARKERS['peripheral']:
        head.append(patt)
        tail.append(patt)

    for patt in MARKERS['infix']:
        mid.append(patt)

    mids = '(%s)' % ('|'.join(mid))
    ends = '(%s)' % ('|'.join(head))
    head_patt = re.compile(f'^{ends}.+{mids}.+')
    tail_patt = re.compile(f'.+{mids}.+{ends}$')

    yield head_patt
    yield tail_patt

    sed = MARKERS['sed']
    sed_patt = '(%s).+(%s).+(%s)' % sed
    yield sed_patt

def markers_finders():
    single_word = r'([^\s]+)'
    for patt in MARKERS['clitic']:
        yield re.compile(single_word + patt)
        yield re.compile(patt + single_word)


def error2str(exc):
    return exc.__class__.__name__ + ': ' + str(exc)


def iterated_lev_dist(a, b, tolerance=5):
    """Calculates edit distance between strings a and b.

    This is a version of the Wagner-Fischer algorithm for calculating
    Levenshtein distance. The core steps are adapted to Pythonic form from code
    listed on the Wikipedia page:
      https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm

    This function short-circuits if
    (1) the difference in string lengths, or
    (2) half the number of characters unique to either string
    exceeds the value given by `tolerance`. The short-circuited output is -1,
    which can be assumed to mean that the edit distance is at least 5.

    Args:
    a, b (str): Strings to compare.
    tolerance (int): The maximum distance we are interested in. This is used to
                     short-circuit the function when the minimum possible cost
                     exceeds this value.
    """
    lenA, lenB = _len(a), _len(b)

    # Optimize away edge cases...
    if a == b:
        return 0

    # Lower bound is length diff, quit if length difference is too big
    if _abs(lenA - lenB) > tolerance:
        return -1

    # Quit early if we know the minimum number of substitutions to make will
    # already be too big
    min_substitutions = _len(_symdif({*a}, {*b})) / 2
    if min_substitutions > tolerance:
        return -1

    # We need 2 arrays, 1 larger than the compared
    slots = _len(b) + 1
    v0 = list(range(slots))
    v1 = [0] * slots

    # print('     ', '  '.join(t))
    # Walking over rows. Each row represents one step along the source string.
    # Last cell of last row gives us the edit distance.
    for i, A in enumerate(a):
        # First cell of each row is for comparison against empty string
        # the cost is always equal to length of source string walked so far
        v1[0] = i + 1

        for j, B in enumerate(b):
            delCost = v0[j + 1] + 1  # j+1 since the first value in the row
                                     # is distance from an empty str
            subCost = v0[j] + (A != B)  # substitution
            insCost = v1[j] + 1  # the insert cost is always at least 1
            v1[j + 1] = _min(delCost, insCost, subCost)

        # When moving to next row, current row
        v0 = _lcopy(v1)
        # print(A, v0)
    return v0[-1]


# def levenshtein_distance(a, b, subweight=1, tolerance=5, sentinel=1000):
#     """Recursively calculates Levenshtein's edit distance between two strings
#
#     Adapted from:
#     https://en.wikipedia.org/wiki/Levenshtein_distance
#
#     Args:
#     a, b: str - Strings
#
#     """
#     i, j = _len(a), _len(b)
#
#     # Upper bound is length diff, quit if length difference is too big
#     if (i - j > tolerance) or (j - i > tolerance):
#         return sentinel
#
#     # Quit early if we know the minimum number of substitutions to make will
#     # already be too big
#     min_substitutions = _len({*a}.symmetric_difference({*b})) // 2
#     if min_substitutions > tolerance:
#         return sentinel
#
#     # Special case where one of the strings is length 0
#     if not (i * j):
#         return _max(i, j)
#
#     subcost = subweight * (a[-1] != b[-1])
#     A, B = a[:-1], b[:-1]
#
#     return _min(
#         levenshtein_distance(A, b) + 1,
#         levenshtein_distance(a, B) + 1,
#         levenshtein_distance(A, B) + subcost
#     )


class Message:
    __slots__ = ['raw', 'id', '_parsed', '_tokens']

    msg_patt = re.compile(r'\[(\d{2}:\d{2})\] <([^>]+)> (.+)')

    def __init__(self, message, msgid):
        self.raw = message  # Original line from source
        self.id = msgid  # Line number, 0-indexed
        self._parsed = None  # False if parsing fails
        self._tokens = None

    @property
    def parsed(self) -> Optional[tuple]:
        if self._parsed != None:
            return self._parsed

        match = self.msg_patt.match(self.raw)
        output = False
        if match:
            timestamp, author, content = match.groups()
            output = (author, timestamp, content)

        self._parsed = output
        return output

    @property
    def author(self):
        if self.parsed:
            return self.parsed[0]
        return None

    @property
    def timestamp(self):
        if self.parsed:
            return self.parsed[1]
        return None

    @property
    def content(self):
        if self.parsed:
            return self.parsed[2]
        return None

    def dumps(self, msgid='', author='', timestamp='', content=''):
        if not self.parsed:
            return self.raw

        return MSG_TEMPLATE.format(msgid=msgid or self.id,
                                   author=author or self.author,
                                   timestamp=timestamp or self.timestamp,
                                   content=content or self.content or self.raw)

    def tokenized(self):
        if self._tokens == None:
            tokens = tokenize(self.content)
            self._tokens = tokens
        return self._tokens

    def token_count(self):
        return len(self.tokenized())

    def __str__(self):
        return self.raw



class Document:
    MSG = 'message'
    TGT = 'target'
    REP = 'repair'

    def __init__(self, docid, sourcefile, annotation='automatic'):
        self.id = docid

        self.source = sourcefile
        self.date, self.language = self.parse_filename(sourcefile)

        self.annotation = annotation

        self.messages = []

    @classmethod
    def parse_filename(cls, filename):
        languages = {
            '-br': 'pt',
            '-cn': 'zh',
            '-es': 'es',
            '-it': 'it',
            '-pl': 'pl',
            '-ru': 'ru',
            '-se': 'sv',
        }
        filename = filename.replace('.txt', '')
        date, channel = filename.split('#', 1)

        date = date[:-1]  # last char is a -
        lang = languages.get(channel[-2:], 'en')
        return date, lang

    def push_message(self, msgobj):
        self.messages.append((self.MSG, msgobj))

    def push_target(self, msgobj, *targetstrs):
        self.messages.append((self.TGT, msgobj, *targetstrs))

    def push_repair(self, msgobj, score, repairstr, msgid, seq):
        self.messages.append((self.REP, msgobj, score, repairstr, msgid, seq))

    def header(self, docid='', annotation='', language='', date='', source='',
               linestart='', lineend=''):

        start, end = None, None
        if self.messages:
            # [1] idx because messages is (mtype, msgobj, ...) tuple
            start = self.messages[0][1].id
            end = self.messages[-1][1].id

        linestart = linestart or start
        lineend = lineend or end

        return DOC_HEADER.format(
            docid=docid or self.id,
            annotation=annotation or self.annotation,
            language=language or self.language,
            date=date or self.date,
            source=source or self.source,
            linestart=start,
            lineend=end
        )

    def dumps(self, iostream=None):
        out = []
        out.append(self.header())

        for mtype, msg, *data in self.messages:
            content = msg.content

            if mtype == self.TGT:
                # data is a list of targetstrs
                content = self.to_target(msg, data)

            elif mtype == self.REP:
                # data = [score, repairstr, msgid, seq]
                content = self.to_repair(msg, *data)

            else:
                pass

            out.append(msg.dumps(content=content))

        out.append(DOC_FOOTER)

        doc = '\n'.join(out)
        if iostream:
            iostream.write(doc)
        else:
            return doc

    @classmethod
    def to_target(cls, msg, targetstrs: list):
        before, tagged, after = '', '', msg.content
        out = []

        for i, tstr in enumerate(targetstrs):
            seq = i + 1
            before, tagged, after = cls.replace_to_tag(
                    cls.TGT, after, tstr, seq=seq)
            out.extend([before, tagged])

        out.append(after)
        return ''.join(out)

    @classmethod
    def to_repair(cls, msg, score, repairstr, msgid, seq):
        before, tagged, after = cls.replace_to_tag(
                cls.REP, msg.content, repairstr,
                message=msgid, seq=seq, confidence=score)
        return f'{before}{tagged}{after}'

    @classmethod
    def replace_to_tag(cls, tag, content, word, **attrs):
        tagged = cls.tagged(tag, word, **attrs)
        pos = content.find(word)
        before = content[:pos]
        after = content[pos + len(word):]
        return before, tagged, after

    @classmethod
    def tagged(cls, tag, content, **attrs):
        attr_str = ''

        if attrs:
            xmlattrs = (f'{k}="{v}"' for k, v in attrs.items())
            attr_str = ' ' + (' '.join(xmlattrs) if attrs else '')

        if content:
            return f'<{tag}{attr_str}>{content}</{tag}>'
        else:
            return f'<{tag}{attr_str} />'



def get_lines(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return [line.rstrip() for line in file.readlines()]

    except OSError as exc:
        estr = error2str(exc)
        print(f'Failed to read file {path} {estr}')


def compare_token(repair, target):
    """Returns 1 if either str is a substr of the other, else Lev dist"""
    if repair in target or target in repair:
        return 1
    return iterated_lev_dist(repair, target)


def compare_tokenized(repairmsg, targetmsg):
    """Compare tokens, returning lowest nonzero edit distance found"""
    nearest = -1

    # Cartesian product of tokens in either msg
    token_pairs = product(repairmsg.tokenized(), targetmsg.tokenized())

    # Reduce to unique pairs of inequivalent tokens
    # This used to change order of r and t but now we keep them...
    tokens = list(set((r, t)# if r < t else (t, r)
                      for r, t in token_pairs
                      if r != t))

    # Get minimum positive dist (distances <0 are edit distances above the
    # short-circuit threshold of our Levenshtein distance function).
    distances = [compare_token(r, t) for r, t in tokens]

    outputs = [(repair, target, dist)
               for ((repair, target), dist) in zip(tokens, distances)
               if dist > 0]
    if not outputs:
        return -1

    # min() will return the tuple with lowest dist
    repair, target, dist = min(outputs, key=lambda tup: tup[2]) # [2] is dist
    return repair, target, dist


def calc_confidence_score(msg, target_candidate):
    distance = iterated_lev_dist
    if not msg.parsed:
        return -1
    if not target_candidate.parsed:
        return -1
    if msg.content.startswith('**'):
        return -1

    score = 0
    target = target_candidate.content
    repair = msg.content

    # Contains repair marker (XOR first and last char) - FIXME: use regex
    score += 3 * msg.content.startswith('*') != msg.content.endswith('*')

    # Repair has short length (weight: 0.5)
    # score += 0.5 if msg.token_count() <= 2 else 0

    msg_edit_dist = distance(msg.content, target_candidate.content)

    # Identical messages are not repairs, halt
    if msg_edit_dist == 0:
        return -1

    # Msg is substring of target candidate (weight: 1)
    if msg.content.startswith(target_candidate.content):
        score += 1

    # Messages that are extremely similar (weight: 5/msg_dist)
    if msg_edit_dist != -1:
        score += _min(5, _len(msg.content)) * (1 / msg_edit_dist)

    # Message is short and repairs a specific word (weight: 1/token_dist)
    if msg_edit_dist > 0:
        # Token-level nonzero edit distance

        compared = compare_tokenized(msg, target_candidate)
        # if min_token_dist > 0:
        if compared != -1:
            repair, target, min_token_dist = compared
            score += 1 / min_token_dist

    # Turn distance, 2/D where D=distance (weight: 0.5)
    # FIXME: Probably works better if we penalize distance...
    score += 0.5 / (msg.id - target_candidate.id)

    # Penalize 0.5 for matching/different author (weight: -0.5)
    score += -0.5 if msg.author != target_candidate.author else 0

    return repair, target, score


def detect_repairs(path, backtrack=REPAIR_WINDOW_SIZE, min_confidence=0):
    msgs = [Message(x, msgid=i) for i, x in enumerate(get_lines(path))]

    for n, msg in enumerate(msgs):
        if n == 0:
            continue

        best_confidence = 0
        m = max(0, n - backtrack)
        back_ctx = msgs[m:n]
        targetmsg, targetstr, repairstr = None, None, None

        for windowmsg in back_ctx:
            tuple_or_minus1 = calc_confidence_score(msg, windowmsg)
            if tuple_or_minus1 == -1:
                continue

            r, t, score = tuple_or_minus1

            # Filter out low confidence
            if score < min_confidence:
                continue

            # Update if this is better than prev likeliest match
            if score > best_confidence:
                targetmsg, repairstr, targetstr = windowmsg, r, t
                best_confidence = score

        if best_confidence > min_confidence:
            back, front = n - backtrack, n + backtrack
            window = msgs[back:front]
            yield window, \
                  msg, repairstr, targetmsg, targetstr, \
                  best_confidence



def main():
    docid = 1

    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    for srcfile in glob(UCC_GLOB, recursive=True):
        if not srcfile.endswith('txt'):
            continue

        name = os.path.basename(srcfile)
        print(name)

        repairs = detect_repairs(srcfile, min_confidence=3.0)  # generator

        for window, r_msg, r_str, t_msg, t_str, score in repairs:

            doc = Document(docid, srcfile, annotation='automatic')

            for msg in window:
                if msg.id == r_msg.id:
                    # def push_repair(self, msgobj, score, repairstr, msgid, seq):
                    doc.push_repair(r_msg, score, r_str, t_msg.id, seq=1)

                elif msg.id == t_msg.id:
                    # def push_target(self, msgobj, *targetstrs):
                    doc.push_target(t_msg, t_str)

                else:
                    doc.push_message(msg)

            outfile = os.path.join(OUTDIR, f'{docid}.xml')
            with open(outfile, 'w', encoding='utf-8') as fs:
                doc.dumps(iostream=fs)

            docid += 1

            # print(f'{name}:{repairmsg.id} ({score}) ({target} -> {repair})')
            # print(f'{targetmsg.id:>5} {targetmsg.raw}')
            # print(f'{repairmsg.id:>5} {repairmsg.raw}')
            # print()


if __name__ == '__main__':
    main()
