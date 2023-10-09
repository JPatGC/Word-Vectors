```python
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import random

from gensim.models import Word2Vec

from sklearn.decomposition import IncrementalPCA   
from sklearn.manifold import TSNE  
```


```python
# read a file you have stored locally
# I added the Hunger Games for simplicity
file = open("carroll-alice.txt", 'r').read()

# first, remove unwanted new line and tab characters from the text
for char in ["\n", "\r", "\d", "\t"]:
    file = file.replace(char, " ")

# check
print(file[:100])
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[3], line 3
          1 # read a file you have stored locally
          2 # I added the Hunger Games for simplicity
    ----> 3 file = open("carroll-alice.txt", 'r').read()
          5 # first, remove unwanted new line and tab characters from the text
          6 for char in ["\n", "\r", "\d", "\t"]:


    File ~/anaconda3/lib/python3.10/site-packages/IPython/core/interactiveshell.py:282, in _modified_open(file, *args, **kwargs)
        275 if file in {0, 1, 2}:
        276     raise ValueError(
        277         f"IPython won't let you open fd={file} by default "
        278         "as it is likely to crash IPython. If you know what you are doing, "
        279         "you can use builtins' open."
        280     )
    --> 282 return io_open(file, *args, **kwargs)


    FileNotFoundError: [Errno 2] No such file or directory: 'carroll-alice.txt'



```python
# read a file you have stored locally
# I added the Hunger Games for simplicity
file = open("carroll-alice.txt", 'r').read()

# first, remove unwanted new line and tab characters from the text
for char in ["\n", "\r", "\d", "\t"]:
    file = file.replace(char, " ")

# check
print(file[:100])
```

    [Alice's Adventures in Wonderland by Lewis Carroll 1865]  CHAPTER I. Down the Rabbit-Hole  Alice was



```python
# this is simplified for demonstration
def sample_clean_text(text: str):
    # step 1: tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # step 2: tokenize each sentence into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # step 3: convert each word to lowercase
    tokenized_text = [[word.lower() for word in sent] for sent in tokenized_sentences]
    
    # return your tokens
    return tokenized_text

# call the function
tokens = sample_clean_text(text = file)

# check
print(tokens[:10])
```

    [['[', 'alice', "'s", 'adventures', 'in', 'wonderland', 'by', 'lewis', 'carroll', '1865', ']', 'chapter', 'i', '.'], ['down', 'the', 'rabbit-hole', 'alice', 'was', 'beginning', 'to', 'get', 'very', 'tired', 'of', 'sitting', 'by', 'her', 'sister', 'on', 'the', 'bank', ',', 'and', 'of', 'having', 'nothing', 'to', 'do', ':', 'once', 'or', 'twice', 'she', 'had', 'peeped', 'into', 'the', 'book', 'her', 'sister', 'was', 'reading', ',', 'but', 'it', 'had', 'no', 'pictures', 'or', 'conversations', 'in', 'it', ',', "'and", 'what', 'is', 'the', 'use', 'of', 'a', 'book', ',', "'", 'thought', 'alice', "'without", 'pictures', 'or', 'conversation', '?', "'"], ['so', 'she', 'was', 'considering', 'in', 'her', 'own', 'mind', '(', 'as', 'well', 'as', 'she', 'could', ',', 'for', 'the', 'hot', 'day', 'made', 'her', 'feel', 'very', 'sleepy', 'and', 'stupid', ')', ',', 'whether', 'the', 'pleasure', 'of', 'making', 'a', 'daisy-chain', 'would', 'be', 'worth', 'the', 'trouble', 'of', 'getting', 'up', 'and', 'picking', 'the', 'daisies', ',', 'when', 'suddenly', 'a', 'white', 'rabbit', 'with', 'pink', 'eyes', 'ran', 'close', 'by', 'her', '.'], ['there', 'was', 'nothing', 'so', 'very', 'remarkable', 'in', 'that', ';', 'nor', 'did', 'alice', 'think', 'it', 'so', 'very', 'much', 'out', 'of', 'the', 'way', 'to', 'hear', 'the', 'rabbit', 'say', 'to', 'itself', ',', "'oh", 'dear', '!'], ['oh', 'dear', '!'], ['i', 'shall', 'be', 'late', '!', "'"], ['(', 'when', 'she', 'thought', 'it', 'over', 'afterwards', ',', 'it', 'occurred', 'to', 'her', 'that', 'she', 'ought', 'to', 'have', 'wondered', 'at', 'this', ',', 'but', 'at', 'the', 'time', 'it', 'all', 'seemed', 'quite', 'natural', ')', ';', 'but', 'when', 'the', 'rabbit', 'actually', 'took', 'a', 'watch', 'out', 'of', 'its', 'waistcoat-pocket', ',', 'and', 'looked', 'at', 'it', ',', 'and', 'then', 'hurried', 'on', ',', 'alice', 'started', 'to', 'her', 'feet', ',', 'for', 'it', 'flashed', 'across', 'her', 'mind', 'that', 'she', 'had', 'never', 'before', 'seen', 'a', 'rabbit', 'with', 'either', 'a', 'waistcoat-pocket', ',', 'or', 'a', 'watch', 'to', 'take', 'out', 'of', 'it', ',', 'and', 'burning', 'with', 'curiosity', ',', 'she', 'ran', 'across', 'the', 'field', 'after', 'it', ',', 'and', 'fortunately', 'was', 'just', 'in', 'time', 'to', 'see', 'it', 'pop', 'down', 'a', 'large', 'rabbit-hole', 'under', 'the', 'hedge', '.'], ['in', 'another', 'moment', 'down', 'went', 'alice', 'after', 'it', ',', 'never', 'once', 'considering', 'how', 'in', 'the', 'world', 'she', 'was', 'to', 'get', 'out', 'again', '.'], ['the', 'rabbit-hole', 'went', 'straight', 'on', 'like', 'a', 'tunnel', 'for', 'some', 'way', ',', 'and', 'then', 'dipped', 'suddenly', 'down', ',', 'so', 'suddenly', 'that', 'alice', 'had', 'not', 'a', 'moment', 'to', 'think', 'about', 'stopping', 'herself', 'before', 'she', 'found', 'herself', 'falling', 'down', 'a', 'very', 'deep', 'well', '.'], ['either', 'the', 'well', 'was', 'very', 'deep', ',', 'or', 'she', 'fell', 'very', 'slowly', ',', 'for', 'she', 'had', 'plenty', 'of', 'time', 'as', 'she', 'went', 'down', 'to', 'look', 'about', 'her', 'and', 'to', 'wonder', 'what', 'was', 'going', 'to', 'happen', 'next', '.']]



```python
model = Word2Vec(tokens,vector_size=100)
```


```python
model.wv.key_to_index
```




    {',': 0,
     'the': 1,
     "'": 2,
     '.': 3,
     'and': 4,
     'to': 5,
     'a': 6,
     'she': 7,
     'i': 8,
     'it': 9,
     'of': 10,
     'said': 11,
     '!': 12,
     'alice': 13,
     'was': 14,
     'you': 15,
     'in': 16,
     'that': 17,
     '--': 18,
     'as': 19,
     'her': 20,
     ':': 21,
     "n't": 22,
     'at': 23,
     '?': 24,
     "'s": 25,
     ';': 26,
     'on': 27,
     'had': 28,
     'with': 29,
     'all': 30,
     'be': 31,
     'for': 32,
     'so': 33,
     'very': 34,
     'they': 35,
     'not': 36,
     'this': 37,
     'but': 38,
     'little': 39,
     'do': 40,
     'he': 41,
     'is': 42,
     'out': 43,
     'what': 44,
     'down': 45,
     'one': 46,
     'up': 47,
     'his': 48,
     'about': 49,
     'would': 50,
     'them': 51,
     'know': 52,
     'there': 53,
     'were': 54,
     'could': 55,
     'have': 56,
     'like': 57,
     'herself': 58,
     'went': 59,
     'again': 60,
     'then': 61,
     'no': 62,
     'queen': 63,
     'if': 64,
     'did': 65,
     'thought': 66,
     'when': 67,
     'or': 68,
     "''": 69,
     'time': 70,
     'me': 71,
     'see': 72,
     'into': 73,
     'off': 74,
     'king': 75,
     'your': 76,
     '*': 77,
     "'m": 78,
     'turtle': 79,
     'began': 80,
     'by': 81,
     'its': 82,
     "'ll": 83,
     'an': 84,
     'my': 85,
     'who': 86,
     ')': 87,
     'mock': 88,
     'hatter': 89,
     '(': 90,
     "'and": 91,
     "'it": 92,
     'quite': 93,
     'gryphon': 94,
     'think': 95,
     'way': 96,
     'how': 97,
     "'you": 98,
     'much': 99,
     'say': 100,
     'their': 101,
     'some': 102,
     'now': 103,
     'first': 104,
     'head': 105,
     'just': 106,
     'more': 107,
     'thing': 108,
     'here': 109,
     'voice': 110,
     'go': 111,
     'are': 112,
     'rabbit': 113,
     'only': 114,
     'got': 115,
     '``': 116,
     'looked': 117,
     'never': 118,
     'which': 119,
     'get': 120,
     "'ve": 121,
     'must': 122,
     'him': 123,
     'mouse': 124,
     'duchess': 125,
     'round': 126,
     'such': 127,
     'tone': 128,
     'came': 129,
     'dormouse': 130,
     'over': 131,
     'other': 132,
     'after': 133,
     'great': 134,
     "'but": 135,
     'any': 136,
     'been': 137,
     "'what": 138,
     'before': 139,
     "'re": 140,
     'back': 141,
     'well': 142,
     'two': 143,
     'cat': 144,
     'can': 145,
     'from': 146,
     'march': 147,
     'last': 148,
     'will': 149,
     'large': 150,
     'long': 151,
     'once': 152,
     'should': 153,
     'come': 154,
     "'that": 155,
     'put': 156,
     'moment': 157,
     'hare': 158,
     'made': 159,
     'nothing': 160,
     'looking': 161,
     'heard': 162,
     'next': 163,
     'things': 164,
     'white': 165,
     'found': 166,
     'right': 167,
     'door': 168,
     'replied': 169,
     'tell': 170,
     'caterpillar': 171,
     "'d": 172,
     'might': 173,
     'dear': 174,
     'eyes': 175,
     'ca': 176,
     'look': 177,
     'make': 178,
     'going': 179,
     'seemed': 180,
     'upon': 181,
     'poor': 182,
     'too': 183,
     'without': 184,
     'yet': 185,
     'rather': 186,
     'soon': 187,
     'course': 188,
     'away': 189,
     'day': 190,
     'three': 191,
     'while': 192,
     'wo': 193,
     'good': 194,
     'took': 195,
     'felt': 196,
     "'oh": 197,
     'shall': 198,
     'added': 199,
     'does': 200,
     'than': 201,
     "'well": 202,
     'same': 203,
     'another': 204,
     'oh': 205,
     'we': 206,
     "'why": 207,
     'getting': 208,
     'minute': 209,
     "'if": 210,
     'find': 211,
     'half': 212,
     'words': 213,
     "'the": 214,
     'wish': 215,
     'ever': 216,
     'cried': 217,
     'take': 218,
     'sort': 219,
     'sure': 220,
     'however': 221,
     'hand': 222,
     'feet': 223,
     'till': 224,
     'being': 225,
     'even': 226,
     'old': 227,
     'tried': 228,
     'curious': 229,
     'anything': 230,
     'house': 231,
     'table': 232,
     'soup': 233,
     'why': 234,
     'something': 235,
     'enough': 236,
     'wonder': 237,
     'court': 238,
     'use': 239,
     'end': 240,
     'asked': 241,
     'eat': 242,
     'question': 243,
     'side': 244,
     'jury': 245,
     'let': 246,
     'bill': 247,
     'spoke': 248,
     'sat': 249,
     'hastily': 250,
     'under': 251,
     'talking': 252,
     'garden': 253,
     'indeed': 254,
     'high': 255,
     'bit': 256,
     'turned': 257,
     "'how": 258,
     'ran': 259,
     'please': 260,
     'near': 261,
     'seen': 262,
     'idea': 263,
     'saying': 264,
     'done': 265,
     'called': 266,
     'am': 267,
     'gave': 268,
     'mad': 269,
     'face': 270,
     "'come": 271,
     'us': 272,
     'through': 273,
     'these': 274,
     "'they": 275,
     'itself': 276,
     'saw': 277,
     'set': 278,
     'hear': 279,
     'anxiously': 280,
     'perhaps': 281,
     'left': 282,
     'beginning': 283,
     'talk': 284,
     'air': 285,
     'both': 286,
     'remember': 287,
     'low': 288,
     'better': 289,
     'knew': 290,
     'ought': 291,
     'trying': 292,
     "'do": 293,
     'baby': 294,
     'room': 295,
     'grow': 296,
     'close': 297,
     'still': 298,
     'game': 299,
     'dance': 300,
     'speak': 301,
     'tea': 302,
     'size': 303,
     'used': 304,
     'gone': 305,
     'always': 306,
     'certainly': 307,
     'people': 308,
     'suddenly': 309,
     "'no": 310,
     'everything': 311,
     'where': 312,
     'sea': 313,
     'far': 314,
     'behind': 315,
     'cats': 316,
     'may': 317,
     'dodo': 318,
     'change': 319,
     'cook': 320,
     'kept': 321,
     'whole': 322,
     'try': 323,
     'afraid': 324,
     'best': 325,
     'arm': 326,
     'pigeon': 327,
     "'we": 328,
     'begin': 329,
     'turning': 330,
     'finished': 331,
     "'then": 332,
     'among': 333,
     'silence': 334,
     'because': 335,
     'chapter': 336,
     'many': 337,
     "'of": 338,
     'suppose': 339,
     'else': 340,
     'deal': 341,
     'hands': 342,
     'every': 343,
     'hardly': 344,
     "'yes": 345,
     'dinah': 346,
     'majesty': 347,
     'pool': 348,
     'waited': 349,
     'growing': 350,
     'tears': 351,
     'hurry': 352,
     'footman': 353,
     'beautiful': 354,
     'glad': 355,
     'makes': 356,
     'gloves': 357,
     'minutes': 358,
     "'not": 359,
     'though': 360,
     'hurried': 361,
     'life': 362,
     'ask': 363,
     "'there": 364,
     'whether': 365,
     'mind': 366,
     'mouth': 367,
     'small': 368,
     'opened': 369,
     'keep': 370,
     'bottle': 371,
     'heads': 372,
     'lessons': 373,
     'sight': 374,
     'word': 375,
     'really': 376,
     "'now": 377,
     'name': 378,
     'walked': 379,
     'those': 380,
     'rest': 381,
     'trial': 382,
     'foot': 383,
     'fan': 384,
     'repeated': 385,
     'having': 386,
     'read': 387,
     'offended': 388,
     'sitting': 389,
     'mean': 390,
     'queer': 391,
     'child': 392,
     'thinking': 393,
     'yourself': 394,
     'soldiers': 395,
     'conversation': 396,
     'children': 397,
     'own': 398,
     'remarked': 399,
     'birds': 400,
     'remark': 401,
     'nearly': 402,
     'witness': 403,
     'continued': 404,
     'key': 405,
     'knave': 406,
     'glass': 407,
     'help': 408,
     'interrupted': 409,
     'hall': 410,
     'either': 411,
     'tail': 412,
     'rate': 413,
     'different': 414,
     'angrily': 415,
     'matter': 416,
     'shook': 417,
     'give': 418,
     'creatures': 419,
     'together': 420,
     'timidly': 421,
     'reason': 422,
     'coming': 423,
     'least': 424,
     'waiting': 425,
     'shouted': 426,
     'few': 427,
     'join': 428,
     'answer': 429,
     'against': 430,
     'believe': 431,
     'sister': 432,
     'puzzled': 433,
     'has': 434,
     'want': 435,
     'meaning': 436,
     'explain': 437,
     'hearts': 438,
     'running': 439,
     'nose': 440,
     'gardeners': 441,
     "'off": 442,
     'mushroom': 443,
     "'she": 444,
     'opportunity': 445,
     "'he": 446,
     'window': 447,
     'whiting': 448,
     'distance': 449,
     'seem': 450,
     'slates': 451,
     'story': 452,
     'turn': 453,
     'changed': 454,
     'five': 455,
     'followed': 456,
     'happen': 457,
     "'in": 458,
     'lying': 459,
     'most': 460,
     'place': 461,
     'work': 462,
     'asleep': 463,
     'fact': 464,
     'top': 465,
     'pig': 466,
     'ready': 467,
     'hard': 468,
     'mine': 469,
     'slowly': 470,
     'watch': 471,
     'william': 472,
     'party': 473,
     'feel': 474,
     'making': 475,
     'eagerly': 476,
     'dry': 477,
     'beg': 478,
     'our': 479,
     'wood': 480,
     'appeared': 481,
     'noticed': 482,
     'play': 483,
     'live': 484,
     'lobsters': 485,
     'serpent': 486,
     'tarts': 487,
     'adventures': 488,
     'oop': 489,
     "'who": 490,
     'hedgehog': 491,
     'tired': 492,
     'fall': 493,
     'deep': 494,
     'listen': 495,
     'lobster': 496,
     'draw': 497,
     'moral': 498,
     'silent': 499,
     'book': 500,
     'twinkle': 501,
     'pleased': 502,
     'song': 503,
     'world': 504,
     'evidence': 505,
     'happened': 506,
     'learn': 507,
     'eye': 508,
     'middle': 509,
     'wondering': 510,
     'history': 511,
     'golden': 512,
     'open': 513,
     'trees': 514,
     'larger': 515,
     'neck': 516,
     'lory': 517,
     'generally': 518,
     "'are": 519,
     'leave': 520,
     'bright': 521,
     'puppy': 522,
     "'as": 523,
     'frightened': 524,
     'surprised': 525,
     'others': 526,
     'feeling': 527,
     "'so": 528,
     'business': 529,
     'grown': 530,
     'kind': 531,
     'fancy': 532,
     'manage': 533,
     'chin': 534,
     'myself': 535,
     "'for": 536,
     'each': 537,
     'sir': 538,
     'hair': 539,
     'begun': 540,
     'stood': 541,
     'dream': 542,
     'goes': 543,
     'shoes': 544,
     'late': 545,
     'repeat': 546,
     'grin': 547,
     'sharp': 548,
     'told': 549,
     'between': 550,
     'cheshire': 551,
     'times': 552,
     'pepper': 553,
     'himself': 554,
     'ear': 555,
     'nice': 556,
     "'would": 557,
     'exclaimed': 558,
     'loud': 559,
     'e': 560,
     'soo': 561,
     'full': 562,
     'chimney': 563,
     'broken': 564,
     'executed': 565,
     'executioner': 566,
     'nobody': 567,
     'bread-and-butter': 568,
     'exactly': 569,
     'liked': 570,
     'everybody': 571,
     'sing': 572,
     'melancholy': 573,
     'politely': 574,
     'sit': 575,
     'trouble': 576,
     'subject': 577,
     'nonsense': 578,
     'trembling': 579,
     'pardon': 580,
     'understand': 581,
     'dreadfully': 582,
     'stop': 583,
     'sneezing': 584,
     'almost': 585,
     'along': 586,
     'croquet': 587,
     'marked': 588,
     'four': 589,
     'somebody': 590,
     'forgotten': 591,
     'youth': 592,
     'fell': 593,
     'hold': 594,
     'written': 595,
     'roof': 596,
     'english': 597,
     'lizard': 598,
     'jumped': 599,
     'sleep': 600,
     'inches': 601,
     'arms': 602,
     "'or": 603,
     'father': 604,
     'leaves': 605,
     'call': 606,
     'finish': 607,
     'pack': 608,
     'impatiently': 609,
     'procession': 610,
     'reply': 611,
     'instantly': 612,
     'faces': 613,
     'confusion': 614,
     'dare': 615,
     'ears': 616,
     'pocket': 617,
     'sadly': 618,
     'chorus': 619,
     'likely': 620,
     "'one": 621,
     'piece': 622,
     'flamingo': 623,
     'knee': 624,
     'hookah': 625,
     'walk': 626,
     'shriek': 627,
     'taking': 628,
     'height': 629,
     'new': 630,
     'write': 631,
     'asking': 632,
     'evening': 633,
     "'when": 634,
     'become': 635,
     'sometimes': 636,
     'interesting': 637,
     'twice': 638,
     'direction': 639,
     'doing': 640,
     'walking': 641,
     'school': 642,
     'moved': 643,
     'aloud': 644,
     'young': 645,
     'temper': 646,
     "'very": 647,
     'sighed': 648,
     'whispered': 649,
     'notice': 650,
     'crowded': 651,
     'means': 652,
     'hot': 653,
     'stay': 654,
     'ten': 655,
     'nervous': 656,
     'altogether': 657,
     'seems': 658,
     'remembered': 659,
     'strange': 660,
     "'to": 661,
     'quietly': 662,
     'sounded': 663,
     'ground': 664,
     'wrong': 665,
     'seven': 666,
     'happens': 667,
     'sha': 668,
     'eggs': 669,
     'across': 670,
     'morning': 671,
     'usual': 672,
     'dropped': 673,
     'tree': 674,
     'curiosity': 675,
     'kid': 676,
     'sudden': 677,
     'surprise': 678,
     'case': 679,
     'shrill': 680,
     'man': 681,
     'important': 682,
     'stand': 683,
     'sleepy': 684,
     'pair': 685,
     "'let": 686,
     'fetch': 687,
     'home': 688,
     'shut': 689,
     'angry': 690,
     'passed': 691,
     'number': 692,
     'waving': 693,
     'sentence': 694,
     'simple': 695,
     'cut': 696,
     'speaking': 697,
     'meant': 698,
     'finger': 699,
     'often': 700,
     'nearer': 701,
     'swam': 702,
     'drew': 703}




```python
model.wv.get_vector("alice", norm=True)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[8], line 1
    ----> 1 model.wv.get_vector("capitol", norm=True)


    File ~/anaconda3/lib/python3.10/site-packages/gensim/models/keyedvectors.py:446, in KeyedVectors.get_vector(self, key, norm)
        422 def get_vector(self, key, norm=False):
        423     """Get the key's vector, as a 1D numpy array.
        424 
        425     Parameters
       (...)
        444 
        445     """
    --> 446     index = self.get_index(key)
        447     if norm:
        448         self.fill_norms()


    File ~/anaconda3/lib/python3.10/site-packages/gensim/models/keyedvectors.py:420, in KeyedVectors.get_index(self, key, default)
        418     return default
        419 else:
    --> 420     raise KeyError(f"Key '{key}' not present")


    KeyError: "Key 'capitol' not present"



```python
model.wv.get_vector("alice", norm=True)
```




    array([-0.03686217,  0.04075047, -0.02609905,  0.02617337,  0.01673777,
           -0.18759605,  0.09894553,  0.2614223 , -0.09730216, -0.09168454,
            0.00405867, -0.19517948, -0.02181011,  0.03577551, -0.00338368,
           -0.06426921,  0.07745586, -0.13957246, -0.00864112, -0.23078014,
            0.10281576,  0.08900964,  0.07079399, -0.07376342, -0.02584271,
            0.00648348, -0.07341146, -0.07265075, -0.11318617,  0.07913384,
            0.08741982,  0.00148013,  0.1415754 , -0.16959698, -0.10675887,
            0.20459963,  0.00858671, -0.10411713, -0.07446089, -0.13197994,
            0.09565642, -0.08987007, -0.01495818,  0.00914126,  0.12350913,
           -0.04709197, -0.1164104 , -0.10110345,  0.07379207,  0.09921322,
            0.04264027, -0.10302304,  0.04286507,  0.00539758, -0.06651596,
            0.03109744,  0.00949783, -0.03115549, -0.15211044,  0.01518367,
            0.04547398, -0.05133945,  0.02015894, -0.0094426 , -0.08805432,
            0.13239507,  0.04276005,  0.08195602, -0.16659416,  0.12755172,
           -0.05051395,  0.04130458,  0.07598905, -0.09608977,  0.21108921,
            0.01136679,  0.03644448,  0.02611546, -0.0666224 ,  0.02925073,
           -0.10074615, -0.09829269, -0.1129761 ,  0.11215609, -0.06625524,
            0.08595908,  0.12751552,  0.1660867 ,  0.14631332,  0.11124389,
            0.22943296,  0.03484311, -0.05786261,  0.0119311 ,  0.20860063,
            0.1363893 ,  0.01028403, -0.09972799,  0.00776146, -0.04126839],
          dtype=float32)




```python
model.wv.get_vector("rabbit", norm=True)
```




    array([-0.03278732,  0.03836986, -0.02572512,  0.02938434,  0.01652416,
           -0.19029613,  0.10335965,  0.267699  , -0.09081123, -0.09065454,
            0.01434399, -0.19408225, -0.01876464,  0.04408117, -0.00334596,
           -0.0676597 ,  0.07262144, -0.13863944, -0.01069522, -0.23522238,
            0.10030277,  0.0859406 ,  0.06588544, -0.07995463, -0.03226305,
            0.0055731 , -0.06462586, -0.07243093, -0.10680491,  0.07477763,
            0.08625028, -0.00092444,  0.13681805, -0.16163835, -0.11253919,
            0.20011403,  0.01579929, -0.10823073, -0.07499035, -0.13143703,
            0.10264411, -0.08512654, -0.02069522,  0.01117011,  0.12615466,
           -0.04736942, -0.11269626, -0.10744844,  0.07131738,  0.10145234,
            0.03853974, -0.10883705,  0.04900645,  0.00788158, -0.06568208,
            0.0282297 ,  0.00416438, -0.03774404, -0.15849482,  0.01429474,
            0.04488612, -0.05235872,  0.02396677, -0.01665852, -0.08825944,
            0.12768921,  0.03759374,  0.07966279, -0.1704343 ,  0.13234726,
           -0.04557783,  0.04047953,  0.07246042, -0.09979641,  0.20752293,
            0.00951862,  0.03550113,  0.0286678 , -0.07365061,  0.03012842,
           -0.10305542, -0.09908503, -0.11168186,  0.10185885, -0.07050596,
            0.08238091,  0.11543577,  0.17064823,  0.15045759,  0.10820654,
            0.22634557,  0.03123348, -0.05313945,  0.01835186,  0.21119314,
            0.13890223,  0.01067711, -0.09420904,  0.00848   , -0.04388001],
          dtype=float32)




```python
model = Word2Vec(tokens,vector_size=500)
```


```python
model.wv.key_to_index
```




    {',': 0,
     'the': 1,
     "'": 2,
     '.': 3,
     'and': 4,
     'to': 5,
     'a': 6,
     'she': 7,
     'i': 8,
     'it': 9,
     'of': 10,
     'said': 11,
     '!': 12,
     'alice': 13,
     'was': 14,
     'you': 15,
     'in': 16,
     'that': 17,
     '--': 18,
     'as': 19,
     'her': 20,
     ':': 21,
     "n't": 22,
     'at': 23,
     '?': 24,
     "'s": 25,
     ';': 26,
     'on': 27,
     'had': 28,
     'with': 29,
     'all': 30,
     'be': 31,
     'for': 32,
     'so': 33,
     'very': 34,
     'they': 35,
     'not': 36,
     'this': 37,
     'but': 38,
     'little': 39,
     'do': 40,
     'he': 41,
     'is': 42,
     'out': 43,
     'what': 44,
     'down': 45,
     'one': 46,
     'up': 47,
     'his': 48,
     'about': 49,
     'would': 50,
     'them': 51,
     'know': 52,
     'there': 53,
     'were': 54,
     'could': 55,
     'have': 56,
     'like': 57,
     'herself': 58,
     'went': 59,
     'again': 60,
     'then': 61,
     'no': 62,
     'queen': 63,
     'if': 64,
     'did': 65,
     'thought': 66,
     'when': 67,
     'or': 68,
     "''": 69,
     'time': 70,
     'me': 71,
     'see': 72,
     'into': 73,
     'off': 74,
     'king': 75,
     'your': 76,
     '*': 77,
     "'m": 78,
     'turtle': 79,
     'began': 80,
     'by': 81,
     'its': 82,
     "'ll": 83,
     'an': 84,
     'my': 85,
     'who': 86,
     ')': 87,
     'mock': 88,
     'hatter': 89,
     '(': 90,
     "'and": 91,
     "'it": 92,
     'quite': 93,
     'gryphon': 94,
     'think': 95,
     'way': 96,
     'how': 97,
     "'you": 98,
     'much': 99,
     'say': 100,
     'their': 101,
     'some': 102,
     'now': 103,
     'first': 104,
     'head': 105,
     'just': 106,
     'more': 107,
     'thing': 108,
     'here': 109,
     'voice': 110,
     'go': 111,
     'are': 112,
     'rabbit': 113,
     'only': 114,
     'got': 115,
     '``': 116,
     'looked': 117,
     'never': 118,
     'which': 119,
     'get': 120,
     "'ve": 121,
     'must': 122,
     'him': 123,
     'mouse': 124,
     'duchess': 125,
     'round': 126,
     'such': 127,
     'tone': 128,
     'came': 129,
     'dormouse': 130,
     'over': 131,
     'other': 132,
     'after': 133,
     'great': 134,
     "'but": 135,
     'any': 136,
     'been': 137,
     "'what": 138,
     'before': 139,
     "'re": 140,
     'back': 141,
     'well': 142,
     'two': 143,
     'cat': 144,
     'can': 145,
     'from': 146,
     'march': 147,
     'last': 148,
     'will': 149,
     'large': 150,
     'long': 151,
     'once': 152,
     'should': 153,
     'come': 154,
     "'that": 155,
     'put': 156,
     'moment': 157,
     'hare': 158,
     'made': 159,
     'nothing': 160,
     'looking': 161,
     'heard': 162,
     'next': 163,
     'things': 164,
     'white': 165,
     'found': 166,
     'right': 167,
     'door': 168,
     'replied': 169,
     'tell': 170,
     'caterpillar': 171,
     "'d": 172,
     'might': 173,
     'dear': 174,
     'eyes': 175,
     'ca': 176,
     'look': 177,
     'make': 178,
     'going': 179,
     'seemed': 180,
     'upon': 181,
     'poor': 182,
     'too': 183,
     'without': 184,
     'yet': 185,
     'rather': 186,
     'soon': 187,
     'course': 188,
     'away': 189,
     'day': 190,
     'three': 191,
     'while': 192,
     'wo': 193,
     'good': 194,
     'took': 195,
     'felt': 196,
     "'oh": 197,
     'shall': 198,
     'added': 199,
     'does': 200,
     'than': 201,
     "'well": 202,
     'same': 203,
     'another': 204,
     'oh': 205,
     'we': 206,
     "'why": 207,
     'getting': 208,
     'minute': 209,
     "'if": 210,
     'find': 211,
     'half': 212,
     'words': 213,
     "'the": 214,
     'wish': 215,
     'ever': 216,
     'cried': 217,
     'take': 218,
     'sort': 219,
     'sure': 220,
     'however': 221,
     'hand': 222,
     'feet': 223,
     'till': 224,
     'being': 225,
     'even': 226,
     'old': 227,
     'tried': 228,
     'curious': 229,
     'anything': 230,
     'house': 231,
     'table': 232,
     'soup': 233,
     'why': 234,
     'something': 235,
     'enough': 236,
     'wonder': 237,
     'court': 238,
     'use': 239,
     'end': 240,
     'asked': 241,
     'eat': 242,
     'question': 243,
     'side': 244,
     'jury': 245,
     'let': 246,
     'bill': 247,
     'spoke': 248,
     'sat': 249,
     'hastily': 250,
     'under': 251,
     'talking': 252,
     'garden': 253,
     'indeed': 254,
     'high': 255,
     'bit': 256,
     'turned': 257,
     "'how": 258,
     'ran': 259,
     'please': 260,
     'near': 261,
     'seen': 262,
     'idea': 263,
     'saying': 264,
     'done': 265,
     'called': 266,
     'am': 267,
     'gave': 268,
     'mad': 269,
     'face': 270,
     "'come": 271,
     'us': 272,
     'through': 273,
     'these': 274,
     "'they": 275,
     'itself': 276,
     'saw': 277,
     'set': 278,
     'hear': 279,
     'anxiously': 280,
     'perhaps': 281,
     'left': 282,
     'beginning': 283,
     'talk': 284,
     'air': 285,
     'both': 286,
     'remember': 287,
     'low': 288,
     'better': 289,
     'knew': 290,
     'ought': 291,
     'trying': 292,
     "'do": 293,
     'baby': 294,
     'room': 295,
     'grow': 296,
     'close': 297,
     'still': 298,
     'game': 299,
     'dance': 300,
     'speak': 301,
     'tea': 302,
     'size': 303,
     'used': 304,
     'gone': 305,
     'always': 306,
     'certainly': 307,
     'people': 308,
     'suddenly': 309,
     "'no": 310,
     'everything': 311,
     'where': 312,
     'sea': 313,
     'far': 314,
     'behind': 315,
     'cats': 316,
     'may': 317,
     'dodo': 318,
     'change': 319,
     'cook': 320,
     'kept': 321,
     'whole': 322,
     'try': 323,
     'afraid': 324,
     'best': 325,
     'arm': 326,
     'pigeon': 327,
     "'we": 328,
     'begin': 329,
     'turning': 330,
     'finished': 331,
     "'then": 332,
     'among': 333,
     'silence': 334,
     'because': 335,
     'chapter': 336,
     'many': 337,
     "'of": 338,
     'suppose': 339,
     'else': 340,
     'deal': 341,
     'hands': 342,
     'every': 343,
     'hardly': 344,
     "'yes": 345,
     'dinah': 346,
     'majesty': 347,
     'pool': 348,
     'waited': 349,
     'growing': 350,
     'tears': 351,
     'hurry': 352,
     'footman': 353,
     'beautiful': 354,
     'glad': 355,
     'makes': 356,
     'gloves': 357,
     'minutes': 358,
     "'not": 359,
     'though': 360,
     'hurried': 361,
     'life': 362,
     'ask': 363,
     "'there": 364,
     'whether': 365,
     'mind': 366,
     'mouth': 367,
     'small': 368,
     'opened': 369,
     'keep': 370,
     'bottle': 371,
     'heads': 372,
     'lessons': 373,
     'sight': 374,
     'word': 375,
     'really': 376,
     "'now": 377,
     'name': 378,
     'walked': 379,
     'those': 380,
     'rest': 381,
     'trial': 382,
     'foot': 383,
     'fan': 384,
     'repeated': 385,
     'having': 386,
     'read': 387,
     'offended': 388,
     'sitting': 389,
     'mean': 390,
     'queer': 391,
     'child': 392,
     'thinking': 393,
     'yourself': 394,
     'soldiers': 395,
     'conversation': 396,
     'children': 397,
     'own': 398,
     'remarked': 399,
     'birds': 400,
     'remark': 401,
     'nearly': 402,
     'witness': 403,
     'continued': 404,
     'key': 405,
     'knave': 406,
     'glass': 407,
     'help': 408,
     'interrupted': 409,
     'hall': 410,
     'either': 411,
     'tail': 412,
     'rate': 413,
     'different': 414,
     'angrily': 415,
     'matter': 416,
     'shook': 417,
     'give': 418,
     'creatures': 419,
     'together': 420,
     'timidly': 421,
     'reason': 422,
     'coming': 423,
     'least': 424,
     'waiting': 425,
     'shouted': 426,
     'few': 427,
     'join': 428,
     'answer': 429,
     'against': 430,
     'believe': 431,
     'sister': 432,
     'puzzled': 433,
     'has': 434,
     'want': 435,
     'meaning': 436,
     'explain': 437,
     'hearts': 438,
     'running': 439,
     'nose': 440,
     'gardeners': 441,
     "'off": 442,
     'mushroom': 443,
     "'she": 444,
     'opportunity': 445,
     "'he": 446,
     'window': 447,
     'whiting': 448,
     'distance': 449,
     'seem': 450,
     'slates': 451,
     'story': 452,
     'turn': 453,
     'changed': 454,
     'five': 455,
     'followed': 456,
     'happen': 457,
     "'in": 458,
     'lying': 459,
     'most': 460,
     'place': 461,
     'work': 462,
     'asleep': 463,
     'fact': 464,
     'top': 465,
     'pig': 466,
     'ready': 467,
     'hard': 468,
     'mine': 469,
     'slowly': 470,
     'watch': 471,
     'william': 472,
     'party': 473,
     'feel': 474,
     'making': 475,
     'eagerly': 476,
     'dry': 477,
     'beg': 478,
     'our': 479,
     'wood': 480,
     'appeared': 481,
     'noticed': 482,
     'play': 483,
     'live': 484,
     'lobsters': 485,
     'serpent': 486,
     'tarts': 487,
     'adventures': 488,
     'oop': 489,
     "'who": 490,
     'hedgehog': 491,
     'tired': 492,
     'fall': 493,
     'deep': 494,
     'listen': 495,
     'lobster': 496,
     'draw': 497,
     'moral': 498,
     'silent': 499,
     'book': 500,
     'twinkle': 501,
     'pleased': 502,
     'song': 503,
     'world': 504,
     'evidence': 505,
     'happened': 506,
     'learn': 507,
     'eye': 508,
     'middle': 509,
     'wondering': 510,
     'history': 511,
     'golden': 512,
     'open': 513,
     'trees': 514,
     'larger': 515,
     'neck': 516,
     'lory': 517,
     'generally': 518,
     "'are": 519,
     'leave': 520,
     'bright': 521,
     'puppy': 522,
     "'as": 523,
     'frightened': 524,
     'surprised': 525,
     'others': 526,
     'feeling': 527,
     "'so": 528,
     'business': 529,
     'grown': 530,
     'kind': 531,
     'fancy': 532,
     'manage': 533,
     'chin': 534,
     'myself': 535,
     "'for": 536,
     'each': 537,
     'sir': 538,
     'hair': 539,
     'begun': 540,
     'stood': 541,
     'dream': 542,
     'goes': 543,
     'shoes': 544,
     'late': 545,
     'repeat': 546,
     'grin': 547,
     'sharp': 548,
     'told': 549,
     'between': 550,
     'cheshire': 551,
     'times': 552,
     'pepper': 553,
     'himself': 554,
     'ear': 555,
     'nice': 556,
     "'would": 557,
     'exclaimed': 558,
     'loud': 559,
     'e': 560,
     'soo': 561,
     'full': 562,
     'chimney': 563,
     'broken': 564,
     'executed': 565,
     'executioner': 566,
     'nobody': 567,
     'bread-and-butter': 568,
     'exactly': 569,
     'liked': 570,
     'everybody': 571,
     'sing': 572,
     'melancholy': 573,
     'politely': 574,
     'sit': 575,
     'trouble': 576,
     'subject': 577,
     'nonsense': 578,
     'trembling': 579,
     'pardon': 580,
     'understand': 581,
     'dreadfully': 582,
     'stop': 583,
     'sneezing': 584,
     'almost': 585,
     'along': 586,
     'croquet': 587,
     'marked': 588,
     'four': 589,
     'somebody': 590,
     'forgotten': 591,
     'youth': 592,
     'fell': 593,
     'hold': 594,
     'written': 595,
     'roof': 596,
     'english': 597,
     'lizard': 598,
     'jumped': 599,
     'sleep': 600,
     'inches': 601,
     'arms': 602,
     "'or": 603,
     'father': 604,
     'leaves': 605,
     'call': 606,
     'finish': 607,
     'pack': 608,
     'impatiently': 609,
     'procession': 610,
     'reply': 611,
     'instantly': 612,
     'faces': 613,
     'confusion': 614,
     'dare': 615,
     'ears': 616,
     'pocket': 617,
     'sadly': 618,
     'chorus': 619,
     'likely': 620,
     "'one": 621,
     'piece': 622,
     'flamingo': 623,
     'knee': 624,
     'hookah': 625,
     'walk': 626,
     'shriek': 627,
     'taking': 628,
     'height': 629,
     'new': 630,
     'write': 631,
     'asking': 632,
     'evening': 633,
     "'when": 634,
     'become': 635,
     'sometimes': 636,
     'interesting': 637,
     'twice': 638,
     'direction': 639,
     'doing': 640,
     'walking': 641,
     'school': 642,
     'moved': 643,
     'aloud': 644,
     'young': 645,
     'temper': 646,
     "'very": 647,
     'sighed': 648,
     'whispered': 649,
     'notice': 650,
     'crowded': 651,
     'means': 652,
     'hot': 653,
     'stay': 654,
     'ten': 655,
     'nervous': 656,
     'altogether': 657,
     'seems': 658,
     'remembered': 659,
     'strange': 660,
     "'to": 661,
     'quietly': 662,
     'sounded': 663,
     'ground': 664,
     'wrong': 665,
     'seven': 666,
     'happens': 667,
     'sha': 668,
     'eggs': 669,
     'across': 670,
     'morning': 671,
     'usual': 672,
     'dropped': 673,
     'tree': 674,
     'curiosity': 675,
     'kid': 676,
     'sudden': 677,
     'surprise': 678,
     'case': 679,
     'shrill': 680,
     'man': 681,
     'important': 682,
     'stand': 683,
     'sleepy': 684,
     'pair': 685,
     "'let": 686,
     'fetch': 687,
     'home': 688,
     'shut': 689,
     'angry': 690,
     'passed': 691,
     'number': 692,
     'waving': 693,
     'sentence': 694,
     'simple': 695,
     'cut': 696,
     'speaking': 697,
     'meant': 698,
     'finger': 699,
     'often': 700,
     'nearer': 701,
     'swam': 702,
     'drew': 703}




```python
model.wv.get_vector("alice", norm=True)
```




    array([ 5.81563488e-02,  6.05684593e-02,  8.36031660e-02,  4.46565636e-02,
           -7.64220115e-03, -8.29128474e-02,  1.47225452e-03,  1.10983536e-01,
            1.35899521e-02,  3.23264636e-02, -1.34411836e-02,  4.57452275e-02,
            5.14668971e-02,  2.42363140e-02,  9.25868563e-03, -6.94569647e-02,
           -4.81288880e-02, -3.59089896e-02, -3.76672000e-02, -3.92338708e-02,
            2.88560744e-02, -1.16525507e-02, -9.68958717e-03, -1.78983882e-02,
            2.76043471e-02,  1.73501857e-02,  3.48861367e-02,  2.03152169e-02,
           -8.16910416e-02,  1.46836729e-03,  1.93284061e-02,  2.87400763e-02,
           -3.54253165e-02, -3.96194495e-02,  2.94630285e-02,  3.38595770e-02,
            2.91289818e-02, -8.37630928e-02, -4.13607582e-02, -1.07876539e-01,
           -6.09402396e-02, -2.12844741e-02, -1.16380706e-01,  3.35453711e-02,
           -4.49228659e-02, -6.68156669e-02, -3.72589156e-02,  1.05298692e-02,
           -1.50391338e-02, -4.59107338e-03,  7.46195810e-03, -1.69040691e-02,
            6.32757368e-03, -1.06410995e-01,  7.47411000e-03, -4.49052788e-02,
            3.37450877e-02, -1.38040846e-02, -8.23370833e-03, -1.83762126e-02,
            3.55242528e-02, -3.43463868e-02,  2.42968295e-02, -3.05524189e-02,
           -4.83353846e-02,  5.90491183e-02, -4.16888949e-03,  3.86438332e-02,
            3.07751913e-02,  1.97278131e-02, -4.45173196e-02,  1.08448602e-02,
           -3.09191272e-03, -5.12515567e-02,  6.45821169e-02,  7.18958080e-02,
           -2.20753644e-02, -2.54512634e-02,  2.96763685e-02,  7.16527551e-02,
            5.36244642e-03,  1.75235048e-02,  1.00700585e-02,  6.26522824e-02,
           -1.11478150e-01,  2.30777301e-02,  1.35809854e-02,  6.49503991e-02,
            2.57562585e-02,  9.02570039e-02,  5.14070224e-03,  1.19854056e-04,
           -6.28371909e-02,  2.01372746e-02,  4.98357713e-02, -3.73834418e-03,
            2.20909305e-02, -1.14647308e-02,  1.88361332e-02, -4.01672833e-02,
           -4.78219315e-02,  1.06608570e-02,  1.15076341e-02,  3.43371145e-02,
            2.00249460e-02,  1.29774893e-02, -1.46130510e-02,  8.46931562e-02,
           -2.29432844e-02,  6.42711297e-02, -2.91515738e-02,  8.54138192e-03,
           -2.20202487e-02,  4.68453728e-02, -5.40363155e-02,  4.87235449e-02,
            1.30413817e-02, -5.33355214e-02,  4.14612610e-03, -3.69741954e-02,
           -1.67859683e-03,  1.78788695e-02,  6.31409436e-02, -4.94064167e-02,
            2.32496019e-02,  5.60548063e-03, -1.01955235e-01,  2.25028433e-02,
           -2.48715803e-02,  3.76804881e-02,  3.90552990e-02,  4.19479385e-02,
           -1.54080708e-02, -7.30202869e-02, -1.61126554e-02,  7.59412572e-02,
           -4.48308885e-02,  2.81897746e-02, -5.61197549e-02, -9.58743095e-02,
            6.57182932e-02, -5.58064580e-02,  1.93369556e-02,  1.56905930e-02,
            3.66135836e-02,  1.04833655e-02,  7.18763545e-02, -4.47894372e-02,
            3.73115838e-02,  3.64513434e-02,  3.84799540e-02,  9.11550410e-03,
           -2.68135462e-02,  8.48022327e-02, -6.78285956e-02,  3.13615687e-02,
            4.24420014e-02, -4.88698250e-04, -2.43344288e-02,  1.73142180e-02,
            3.10123190e-02, -1.89320359e-03, -7.12397471e-02, -2.57877689e-02,
           -9.49023589e-02,  5.60894348e-02,  7.66496882e-02,  5.69812246e-02,
            1.23459212e-02,  5.35146594e-02,  2.79296283e-03,  5.13157323e-02,
            1.43613503e-03,  4.54441383e-02,  3.40836681e-02, -4.59323190e-02,
           -3.70596838e-03, -3.08518000e-02, -2.60879062e-02,  8.32384154e-02,
           -5.33418208e-02,  6.57386482e-02,  1.41929919e-02, -4.81529953e-03,
           -2.85897125e-02, -1.39517421e-02, -4.12790999e-02,  5.31858839e-02,
            9.93235968e-03,  7.15036616e-02,  7.25143179e-02, -7.94334635e-02,
            1.27650900e-02,  6.90907380e-03,  7.22882512e-04, -1.06201712e-02,
           -2.83616167e-02, -1.53283300e-02,  3.93000096e-02,  2.56735440e-02,
           -2.37990590e-03,  2.95742811e-03,  7.12617338e-02,  6.33424595e-02,
            2.06068214e-02,  2.47906297e-02, -3.14795412e-03, -3.06970533e-02,
           -7.34556988e-02, -2.51020268e-02,  2.61398789e-04, -4.83368188e-02,
           -2.18628924e-02, -3.36787626e-02, -1.14825796e-02, -9.06336233e-02,
           -1.85053740e-02, -2.14417512e-03, -5.22321165e-02, -5.41928746e-02,
            2.89208023e-03,  6.34903386e-02,  3.39680091e-02,  1.52537450e-02,
            1.60160277e-03, -1.21582439e-02,  4.34328355e-02, -5.73462807e-02,
           -4.16329969e-03,  1.94418896e-02,  1.38796754e-02, -2.76689157e-02,
           -1.84508059e-02, -9.38392151e-03, -1.16528319e-02,  2.49026855e-03,
           -1.33021938e-04, -7.11391568e-02,  6.97712749e-02, -2.36007813e-02,
           -3.09982244e-02, -4.81531247e-02, -1.05462320e-01,  5.22050336e-02,
            1.64544396e-02, -1.83854923e-02, -2.91684195e-02,  1.46263791e-02,
            5.51277287e-02,  2.99658328e-02,  6.00940268e-03, -9.27007645e-02,
            6.62699621e-03, -3.43948975e-02, -6.14797138e-02,  3.42983603e-02,
            1.30141806e-02,  5.73265925e-02, -1.40872598e-01,  4.02021892e-02,
           -1.99604984e-02,  1.42928585e-02, -4.01457176e-02, -4.98943124e-03,
           -2.27961484e-02, -4.05164585e-02, -7.03772753e-02,  6.73018470e-02,
           -8.50416720e-02,  2.56669112e-02, -3.79641689e-02,  3.18538100e-02,
            6.48340136e-02, -1.19931269e-02, -3.19596641e-02,  1.19632930e-02,
           -4.91159149e-02, -6.76161051e-02, -3.26582380e-02,  4.91220765e-02,
            3.10593657e-02, -2.34286711e-02, -5.41642047e-02, -4.64901216e-02,
            1.06981896e-01, -4.71717268e-02,  1.34636578e-03, -2.51939539e-02,
            6.63891733e-02, -1.46767003e-02, -2.77344212e-02,  1.34560674e-01,
            7.93848559e-03, -3.48224603e-02,  1.76959075e-02,  5.15949950e-02,
            5.22898100e-02, -1.03540592e-01,  9.41799767e-03, -4.62852865e-02,
           -6.27094954e-02, -1.70926489e-02,  1.88685786e-02, -3.63570713e-02,
            9.53693409e-03, -4.95361611e-02, -1.11898389e-02,  3.18370350e-02,
           -1.06447078e-02, -1.62384752e-02, -2.13999990e-02, -3.89756486e-02,
            7.49391317e-02,  4.28798646e-02,  1.23377144e-03, -3.49145476e-03,
            1.40800066e-02, -4.66394685e-02,  4.37914841e-02, -8.08669720e-03,
            7.65413940e-02,  5.20870322e-03,  4.99004219e-03, -4.10141237e-02,
           -3.42194028e-02,  1.57526438e-03, -3.33265178e-02, -4.19447571e-02,
            2.96187773e-02,  1.92825999e-02,  6.70464560e-02,  3.19120288e-02,
            6.90976232e-02,  3.80627885e-02, -1.94888394e-02,  3.99542646e-03,
           -3.29410098e-02,  4.41653840e-02, -5.75230159e-02, -6.27065673e-02,
            2.96819431e-04,  2.76427753e-02,  8.34563076e-02, -9.20673609e-02,
           -1.48065509e-02, -5.33290654e-02, -1.18782490e-01,  1.94233857e-04,
            4.63927444e-03,  2.65329704e-02,  3.28861065e-02, -2.42727380e-02,
           -1.10293878e-02,  7.60721415e-02, -4.64409739e-02,  6.76897988e-02,
           -3.69156227e-02, -7.36985309e-03, -4.28822339e-02, -1.71669871e-02,
            1.75613146e-02,  2.12182719e-02, -3.39980088e-02,  3.23626250e-02,
           -5.37241884e-02, -3.15002799e-02,  2.12969929e-02, -2.30566878e-02,
           -8.09291080e-02,  5.32939471e-02,  2.66868789e-02, -4.34216894e-02,
            4.65721823e-03, -8.52467492e-03, -3.38013507e-02, -1.08994469e-02,
            2.08753301e-03,  1.07946368e-02, -4.15174961e-02, -4.85979812e-03,
            4.04815264e-02, -3.31931412e-02,  5.50523065e-02,  3.07502933e-02,
            1.39152985e-02, -3.47368629e-03,  2.45001093e-02,  5.59881330e-02,
            5.88327013e-02, -2.87846942e-02, -2.47436352e-02, -5.51350638e-02,
           -5.43798879e-02,  1.64280199e-02,  7.19398633e-02,  6.20642416e-02,
           -3.04901693e-02,  7.09823333e-03, -1.33122671e-02,  7.44235590e-02,
           -5.47242984e-02,  6.89191790e-03, -3.47565562e-02, -1.17166705e-01,
           -1.21897412e-02, -4.51288261e-02, -2.16655359e-02,  5.98397069e-02,
            1.90797672e-02, -8.27171002e-03, -3.91735248e-02, -4.01597917e-02,
           -9.60858632e-03, -4.55100164e-02,  1.21082854e-03,  3.40748951e-02,
            7.39518106e-02, -4.02172506e-02,  2.78354771e-02, -2.33595800e-02,
           -8.51683412e-03,  1.93498265e-02, -3.66306230e-02,  3.21660973e-02,
           -7.06462488e-02,  2.26363204e-02, -1.11122243e-02, -4.06973474e-02,
           -5.80910370e-02, -2.62507685e-02, -2.91613191e-02, -7.50377961e-03,
           -1.64420642e-02, -1.03059290e-02, -1.61938909e-02,  8.18281993e-03,
           -6.28052559e-03, -4.74076532e-02, -3.01649217e-02,  3.03146280e-02,
           -5.61215030e-03, -3.91718261e-02,  4.04827483e-03, -4.71903905e-02,
            1.48155078e-01,  7.76124885e-03,  5.11305891e-02,  2.26081684e-02,
            2.31957510e-02, -1.94883272e-02,  3.57056484e-02,  1.40398303e-02,
           -7.91198015e-03, -5.25059886e-02, -3.78349749e-03, -5.52188270e-02,
            9.27254558e-03, -4.12189290e-02,  7.31266988e-03, -4.07992564e-02,
            1.79404616e-02, -1.99812260e-02,  1.85783468e-02,  3.32785137e-02,
           -3.65674384e-02, -1.54545195e-02, -7.02645211e-03, -7.98384547e-02,
           -5.17610721e-02,  7.26506263e-02, -6.49622455e-02,  2.27997880e-02,
            7.72240609e-02, -3.76761183e-02,  6.55859783e-02,  7.63054416e-02,
            6.92150369e-02, -1.36906495e-02, -6.16768859e-02, -3.85827273e-02,
           -4.83691804e-02, -1.10990308e-01, -7.11732507e-02,  6.17923662e-02,
            1.33479405e-02,  2.62260400e-02, -2.22285669e-02,  5.83125316e-02,
            2.23852843e-02,  2.07653764e-04, -2.24580104e-03, -3.09186261e-02,
            5.22041395e-02, -3.23247910e-02,  2.00027470e-02,  6.22945055e-02,
           -1.73163675e-02, -5.59667312e-02, -5.15584201e-02, -4.72623520e-02],
          dtype=float32)




```python
model.wv.similarity('alice', 'girl')
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[14], line 1
    ----> 1 model.wv.similarity('alice', 'girl')


    File ~/anaconda3/lib/python3.10/site-packages/gensim/models/keyedvectors.py:1234, in KeyedVectors.similarity(self, w1, w2)
       1218 def similarity(self, w1, w2):
       1219     """Compute cosine similarity between two keys.
       1220 
       1221     Parameters
       (...)
       1232 
       1233     """
    -> 1234     return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))


    File ~/anaconda3/lib/python3.10/site-packages/gensim/models/keyedvectors.py:403, in KeyedVectors.__getitem__(self, key_or_keys)
        389 """Get vector representation of `key_or_keys`.
        390 
        391 Parameters
       (...)
        400 
        401 """
        402 if isinstance(key_or_keys, _KEY_TYPES):
    --> 403     return self.get_vector(key_or_keys)
        405 return vstack([self.get_vector(key) for key in key_or_keys])


    File ~/anaconda3/lib/python3.10/site-packages/gensim/models/keyedvectors.py:446, in KeyedVectors.get_vector(self, key, norm)
        422 def get_vector(self, key, norm=False):
        423     """Get the key's vector, as a 1D numpy array.
        424 
        425     Parameters
       (...)
        444 
        445     """
    --> 446     index = self.get_index(key)
        447     if norm:
        448         self.fill_norms()


    File ~/anaconda3/lib/python3.10/site-packages/gensim/models/keyedvectors.py:420, in KeyedVectors.get_index(self, key, default)
        418     return default
        419 else:
    --> 420     raise KeyError(f"Key '{key}' not present")


    KeyError: "Key 'girl' not present"



```python
model.wv.most_similar('alice')
```




    [('it', 0.9998971819877625),
     (':', 0.9998888969421387),
     ('but', 0.9998868107795715),
     (';', 0.999886155128479),
     ('that', 0.9998844861984253),
     ('very', 0.999880850315094),
     ('when', 0.9998801350593567),
     ('so', 0.9998775124549866),
     ('to', 0.9998762011528015),
     ('he', 0.9998740553855896)]




```python
model.wv.similarity('alice', 'it')
```




    0.9998971




```python
model.wv.similarity('alice', 'he')
```




    0.9998741




```python
model.wv.similarity('alice', 'but')
```




    0.9998867




```python
model.wv.similarity('alice', 'it')
```




    0.9998971




```python
def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(model)
```


```python
def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.12.1.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<div>                            <div id="dbdb2535-5dad-4a76-bd4a-368fbc6f57cf" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("dbdb2535-5dad-4a76-bd4a-368fbc6f57cf")) {                    Plotly.newPlot(                        "dbdb2535-5dad-4a76-bd4a-368fbc6f57cf",                        [{"mode":"text","text":[",","the","'",".","and","to","a","she","i","it","of","said","!","alice","was","you","in","that","--","as","her",":","n't","at","?","'s",";","on","had","with","all","be","for","so","very","they","not","this","but","little","do","he","is","out","what","down","one","up","his","about","would","them","know","there","were","could","have","like","herself","went","again","then","no","queen","if","did","thought","when","or","''","time","me","see","into","off","king","your","*","'m","turtle","began","by","its","'ll","an","my","who",")","mock","hatter","(","'and","'it","quite","gryphon","think","way","how","'you","much","say","their","some","now","first","head","just","more","thing","here","voice","go","are","rabbit","only","got","``","looked","never","which","get","'ve","must","him","mouse","duchess","round","such","tone","came","dormouse","over","other","after","great","'but","any","been","'what","before","'re","back","well","two","cat","can","from","march","last","will","large","long","once","should","come","'that","put","moment","hare","made","nothing","looking","heard","next","things","white","found","right","door","replied","tell","caterpillar","'d","might","dear","eyes","ca","look","make","going","seemed","upon","poor","too","without","yet","rather","soon","course","away","day","three","while","wo","good","took","felt","'oh","shall","added","does","than","'well","same","another","oh","we","'why","getting","minute","'if","find","half","words","'the","wish","ever","cried","take","sort","sure","however","hand","feet","till","being","even","old","tried","curious","anything","house","table","soup","why","something","enough","wonder","court","use","end","asked","eat","question","side","jury","let","bill","spoke","sat","hastily","under","talking","garden","indeed","high","bit","turned","'how","ran","please","near","seen","idea","saying","done","called","am","gave","mad","face","'come","us","through","these","'they","itself","saw","set","hear","anxiously","perhaps","left","beginning","talk","air","both","remember","low","better","knew","ought","trying","'do","baby","room","grow","close","still","game","dance","speak","tea","size","used","gone","always","certainly","people","suddenly","'no","everything","where","sea","far","behind","cats","may","dodo","change","cook","kept","whole","try","afraid","best","arm","pigeon","'we","begin","turning","finished","'then","among","silence","because","chapter","many","'of","suppose","else","deal","hands","every","hardly","'yes","dinah","majesty","pool","waited","growing","tears","hurry","footman","beautiful","glad","makes","gloves","minutes","'not","though","hurried","life","ask","'there","whether","mind","mouth","small","opened","keep","bottle","heads","lessons","sight","word","really","'now","name","walked","those","rest","trial","foot","fan","repeated","having","read","offended","sitting","mean","queer","child","thinking","yourself","soldiers","conversation","children","own","remarked","birds","remark","nearly","witness","continued","key","knave","glass","help","interrupted","hall","either","tail","rate","different","angrily","matter","shook","give","creatures","together","timidly","reason","coming","least","waiting","shouted","few","join","answer","against","believe","sister","puzzled","has","want","meaning","explain","hearts","running","nose","gardeners","'off","mushroom","'she","opportunity","'he","window","whiting","distance","seem","slates","story","turn","changed","five","followed","happen","'in","lying","most","place","work","asleep","fact","top","pig","ready","hard","mine","slowly","watch","william","party","feel","making","eagerly","dry","beg","our","wood","appeared","noticed","play","live","lobsters","serpent","tarts","adventures","oop","'who","hedgehog","tired","fall","deep","listen","lobster","draw","moral","silent","book","twinkle","pleased","song","world","evidence","happened","learn","eye","middle","wondering","history","golden","open","trees","larger","neck","lory","generally","'are","leave","bright","puppy","'as","frightened","surprised","others","feeling","'so","business","grown","kind","fancy","manage","chin","myself","'for","each","sir","hair","begun","stood","dream","goes","shoes","late","repeat","grin","sharp","told","between","cheshire","times","pepper","himself","ear","nice","'would","exclaimed","loud","e","soo","full","chimney","broken","executed","executioner","nobody","bread-and-butter","exactly","liked","everybody","sing","melancholy","politely","sit","trouble","subject","nonsense","trembling","pardon","understand","dreadfully","stop","sneezing","almost","along","croquet","marked","four","somebody","forgotten","youth","fell","hold","written","roof","english","lizard","jumped","sleep","inches","arms","'or","father","leaves","call","finish","pack","impatiently","procession","reply","instantly","faces","confusion","dare","ears","pocket","sadly","chorus","likely","'one","piece","flamingo","knee","hookah","walk","shriek","taking","height","new","write","asking","evening","'when","become","sometimes","interesting","twice","direction","doing","walking","school","moved","aloud","young","temper","'very","sighed","whispered","notice","crowded","means","hot","stay","ten","nervous","altogether","seems","remembered","strange","'to","quietly","sounded","ground","wrong","seven","happens","sha","eggs","across","morning","usual","dropped","tree","curiosity","kid","sudden","surprise","case","shrill","man","important","stand","sleepy","pair","'let","fetch","home","shut","angry","passed","number","waving","sentence","simple","cut","speaking","meant","finger","often","nearer","swam","drew"],"x":[45.369712829589844,45.81214141845703,44.29597091674805,45.22023391723633,45.58690643310547,46.56635284423828,46.580257415771484,45.788787841796875,45.73832321166992,45.252235412597656,46.121307373046875,42.30485916137695,45.05040740966797,43.68210220336914,45.438594818115234,46.15574264526367,46.913082122802734,45.40434646606445,45.75727081298828,44.56209945678711,46.22547912597656,46.08464050292969,45.95565414428711,45.33683395385742,42.808780670166016,45.88640594482422,45.238494873046875,46.53759765625,44.81977844238281,45.774593353271484,43.77806854248047,43.23421096801758,44.956947326660156,43.64862060546875,45.27992630004883,46.411651611328125,43.63924789428711,44.80474853515625,44.36727523803711,45.01787185668945,43.257080078125,44.425994873046875,43.38520431518555,45.64128494262695,43.51250076293945,44.28081130981445,44.896732330322266,44.22770309448242,46.48990249633789,44.131134033203125,44.212711334228516,42.3911247253418,35.92177963256836,42.299129486083984,45.41872024536133,40.84291076660156,44.5227165222168,37.156890869140625,41.83880615234375,39.96406555175781,40.74981689453125,41.80729675292969,41.36082458496094,32.70490646362305,43.4693603515625,40.16628646850586,39.52646255493164,44.81221389770508,45.54143524169922,37.27839660644531,36.003204345703125,42.687744140625,41.56317138671875,43.175350189208984,43.09180450439453,33.50597381591797,42.97067642211914,47.693172454833984,36.75971221923828,40.191829681396484,40.341304779052734,41.472869873046875,44.70150375366211,42.30793762207031,41.13508224487305,44.01131820678711,42.87299728393555,39.53142166137695,36.71601867675781,33.992340087890625,39.73439407348633,40.67013168334961,33.26478576660156,41.21513366699219,32.202049255371094,31.085920333862305,38.44852066040039,39.427066802978516,37.635650634765625,35.21333312988281,37.774620056152344,42.83750534057617,42.43077850341797,39.12255859375,37.26076126098633,32.45573425292969,42.59550094604492,42.81990051269531,41.60746383666992,40.121028900146484,39.1571159362793,39.03608703613281,39.37916946411133,39.63223648071289,37.87180709838867,32.38531494140625,41.09007263183594,35.4709587097168,35.08747863769531,40.834983825683594,40.61033248901367,36.9115104675293,37.19017791748047,36.099666595458984,27.514650344848633,23.52210235595703,38.13796615600586,32.079532623291016,30.394697189331055,35.53483581542969,36.3936767578125,33.149784088134766,38.944400787353516,33.56678009033203,28.49481201171875,34.82748031616211,36.28982925415039,28.124399185180664,21.9547176361084,38.41327667236328,39.717464447021484,33.13811111450195,29.27229118347168,36.36769104003906,28.633893966674805,34.6678581237793,38.856101989746094,36.73320007324219,25.312591552734375,28.529075622558594,35.58433532714844,31.357585906982422,35.396156311035156,25.3763370513916,26.581331253051758,26.722705841064453,31.653396606445312,20.35428810119629,31.957599639892578,29.63590431213379,26.836971282958984,33.86178207397461,28.36876678466797,24.8139705657959,24.957809448242188,34.043617248535156,20.039060592651367,33.05703353881836,28.992076873779297,24.00743293762207,20.33587074279785,14.32620620727539,38.846351623535156,17.200204849243164,19.219884872436523,30.335617065429688,21.006305694580078,13.9008207321167,22.154977798461914,28.106456756591797,22.813182830810547,34.96220016479492,25.429784774780273,28.128154754638672,28.930513381958008,20.849899291992188,18.417207717895508,22.636205673217773,21.56710433959961,26.141878128051758,17.526226043701172,29.01145362854004,21.848464965820312,16.539827346801758,27.909591674804688,29.76924705505371,15.861328125,15.055474281311035,21.061206817626953,20.472013473510742,25.83268928527832,27.777767181396484,1.0349624156951904,22.965543746948242,22.383441925048828,12.130393981933594,29.183712005615234,12.965145111083984,21.560991287231445,18.28082847595215,17.5113468170166,14.457097053527832,21.22577667236328,17.073501586914062,19.90140724182129,13.953093528747559,26.009824752807617,8.668347358703613,22.113162994384766,16.75850486755371,14.371597290039062,10.809525489807129,13.092713356018066,13.260574340820312,22.276891708374023,17.84665298461914,20.31549835205078,15.274800300598145,12.116674423217773,12.986420631408691,21.915586471557617,26.481069564819336,11.451196670532227,23.08026885986328,14.646925926208496,13.725719451904297,21.38515853881836,16.7439022064209,-5.491626262664795,9.204761505126953,13.370001792907715,0.170617938041687,14.026459693908691,10.540868759155273,16.513168334960938,13.69015884399414,16.993778228759766,17.372230529785156,10.200393676757812,15.360696792602539,9.778502464294434,13.483809471130371,13.718648910522461,18.103721618652344,10.608481407165527,14.022061347961426,-4.088041305541992,10.990694999694824,5.745110511779785,12.043121337890625,14.701579093933105,14.548285484313965,10.209062576293945,5.656851768493652,18.734752655029297,14.125423431396484,8.6827392578125,9.695938110351562,14.764492988586426,15.327664375305176,-6.209022521972656,3.5859885215759277,13.141953468322754,12.91230297088623,11.09091854095459,11.763291358947754,11.854082107543945,9.51931095123291,2.059436798095703,10.86523151397705,11.497709274291992,12.667770385742188,7.361265659332275,14.090679168701172,10.488232612609863,1.6925363540649414,8.335835456848145,6.956115245819092,9.838567733764648,13.575325965881348,16.240079879760742,11.373401641845703,11.8733491897583,9.05991268157959,5.062569618225098,14.892931938171387,12.730701446533203,12.051461219787598,12.106244087219238,-11.953003883361816,3.901731491088867,6.6521711349487305,3.12662935256958,10.569903373718262,8.127450942993164,11.203287124633789,15.113813400268555,10.074599266052246,13.0451078414917,12.683459281921387,-27.525615692138672,13.582894325256348,8.322070121765137,-23.998069763183594,16.17533302307129,-8.367823600769043,12.255926132202148,10.52591323852539,12.744085311889648,15.251370429992676,-17.32449722290039,11.796390533447266,13.648446083068848,5.2073822021484375,-5.97107458114624,3.312062978744507,1.6565250158309937,-19.20940399169922,-0.1352374106645584,-12.328226089477539,0.8324699997901917,3.8479697704315186,0.868157684803009,0.8784403204917908,-4.384586334228516,10.865646362304688,-42.50275802612305,7.680481433868408,-10.73212718963623,8.746033668518066,11.585476875305176,17.80575942993164,-12.86352825164795,8.799312591552734,8.046060562133789,-18.25029182434082,11.420614242553711,11.500378608703613,-20.730863571166992,-0.4575422704219818,2.324028968811035,-9.161578178405762,-10.049345970153809,-11.425773620605469,17.074485778808594,3.7647342681884766,-0.38628730177879333,12.962181091308594,11.88335132598877,-26.264162063598633,7.962850570678711,1.9135792255401611,-37.03950881958008,-10.305011749267578,6.163214683532715,-1.1490298509597778,-4.647487163543701,-15.976141929626465,-7.299473285675049,-3.523533582687378,5.713772773742676,4.611395359039307,-24.506633758544922,8.97603988647461,-23.69694709777832,2.720930576324463,5.824027061462402,4.943123817443848,-4.2639851570129395,3.1129188537597656,-5.401490211486816,-1.4976879358291626,2.8859314918518066,-5.087196350097656,8.178200721740723,4.1779608726501465,10.933669090270996,4.21347713470459,11.076274871826172,10.900341033935547,-3.886399745941162,1.4076706171035767,2.4718594551086426,-5.010892868041992,-13.328935623168945,-8.838549613952637,-18.05598258972168,-16.67192840576172,-3.8466484546661377,-10.049356460571289,-29.381507873535156,-10.321305274963379,-14.049229621887207,-13.086833953857422,-6.600318908691406,6.079710960388184,-14.86511516571045,10.149664878845215,-12.9579439163208,7.535301208496094,-16.118545532226562,-16.909378051757812,11.51578140258789,-18.8548526763916,-11.09676456451416,-9.808706283569336,1.17638099193573,7.604552745819092,-13.499284744262695,-7.731906414031982,0.4871060848236084,-11.548788070678711,-22.184986114501953,-11.508566856384277,9.738532066345215,-9.763195991516113,-34.640647888183594,9.433394432067871,-9.578627586364746,-7.783929824829102,-0.09520496428012848,-41.13261413574219,-21.50289535522461,-14.435637474060059,-15.55221176147461,7.225627422332764,-29.094146728515625,-13.19569206237793,-31.271459579467773,-17.572734832763672,-15.804851531982422,-12.523775100708008,-49.43691635131836,-12.283858299255371,-16.331880569458008,-13.471302032470703,-46.0782470703125,-19.060014724731445,-16.242612838745117,-46.91164016723633,-6.864987850189209,-7.838403701782227,7.039236545562744,-14.839387893676758,-15.815909385681152,-15.133221626281738,-12.616661071777344,-13.159966468811035,-11.857345581054688,-26.72359848022461,11.589248657226562,11.687570571899414,-35.65443420410156,-7.243542194366455,-21.209484100341797,-19.096210479736328,-20.632631301879883,-0.27524253726005554,-29.47719955444336,-41.10171890258789,-6.077059268951416,-15.719223022460938,-13.977547645568848,-14.478575706481934,-17.07017707824707,-16.812847137451172,-14.499651908874512,-7.25433349609375,-16.569782257080078,0.5061272978782654,-33.77781677246094,-26.811418533325195,-40.7543830871582,-9.520872116088867,-23.28261375427246,-24.949281692504883,-48.16171646118164,-19.594648361206055,-22.552955627441406,-3.8221867084503174,-48.37302017211914,-49.56916427612305,-13.909494400024414,-9.551239013671875,-32.6701774597168,-46.292396545410156,-22.671680450439453,-9.003144264221191,-49.62876892089844,-21.92196273803711,-44.12220001220703,11.189037322998047,-11.896227836608887,-25.314401626586914,-28.030006408691406,-12.522529602050781,-6.271556854248047,-29.795223236083984,6.8526387214660645,-14.594045639038086,-12.625598907470703,-41.1019172668457,-4.287613391876221,-19.731739044189453,-34.718204498291016,-6.882692337036133,-34.49176025390625,-46.988948822021484,-42.245445251464844,-40.189884185791016,-30.028615951538086,-11.581276893615723,-16.944683074951172,-43.34846115112305,-14.130572319030762,-46.694175720214844,-12.0269193649292,-22.373252868652344,-11.329276084899902,-30.164335250854492,-15.191333770751953,-16.51535987854004,-23.279876708984375,-43.7609748840332,-28.6446533203125,-14.046201705932617,-7.794536113739014,-13.229747772216797,-27.175626754760742,-12.17706298828125,1.5021272897720337,-9.205023765563965,-33.824867248535156,-21.403820037841797,-18.23487091064453,-46.703338623046875,-12.910554885864258,-25.79193878173828,-6.748281478881836,-43.4128303527832,-36.80534362792969,-37.21710968017578,6.8748626708984375,-34.224517822265625,-24.71342658996582,-40.829105377197266,-36.69871520996094,-43.596187591552734,-48.552425384521484,-34.948753356933594,10.021098136901855,-20.14272689819336,-18.788604736328125,-12.87781047821045,-24.201597213745117,-46.014671325683594,-47.99325942993164,-44.866722106933594,-18.57699966430664,-15.421066284179688,-44.94416427612305,-9.257405281066895,-15.75330924987793,-30.149314880371094,-46.53160858154297,-22.064666748046875,-38.94523620605469,-18.815603256225586,-33.71461486816406,-43.988616943359375,-47.79418182373047,-37.87586212158203,-22.847341537475586,-42.73715591430664,-33.67559814453125,-20.017597198486328,-24.297462463378906,-29.407894134521484,-49.19294357299805,-3.1336827278137207,-31.927682876586914,-43.7039909362793,-19.066951751708984,-8.866171836853027,-36.29965591430664,-33.29953384399414,-32.45809555053711,-46.02896499633789,-46.300968170166016,-25.55354881286621,-15.330912590026855,-29.119001388549805,-37.65223693847656,-14.754059791564941,-12.496455192565918,-37.218563079833984,-38.31204605102539,-13.739474296569824,-45.15097427368164,-11.722110748291016,-47.35334014892578,-21.692110061645508,4.213080883026123,-43.66401672363281,-16.993267059326172,-40.88380813598633,-24.606176376342773,-47.15142059326172,-19.55022621154785,-45.98691177368164,-44.42142868041992,-27.744503021240234,-18.698318481445312,-49.51339340209961,-41.581748962402344,-27.325143814086914,-22.706371307373047,-47.4561767578125,-33.19873046875,-40.67142868041992,-46.09330368041992,-44.142093658447266,-35.5826530456543,-8.402677536010742,-36.58921432495117,-42.0673828125,-29.748119354248047,-17.267343521118164,-16.439075469970703,-13.939172744750977,-43.185794830322266,-35.84879684448242,-34.27726364135742,-31.590139389038086,-49.1880989074707,-38.04056930541992,-49.49607467651367,-49.48589324951172,-32.69152069091797,-37.337074279785156,-8.242643356323242,-36.77099609375,-44.48446273803711,-39.95988464355469,-42.5688591003418,-29.359542846679688,-43.848052978515625,-23.111116409301758,-45.905982971191406,-33.255470275878906,-16.323543548583984,-44.94277572631836,-20.238725662231445,-30.99066734313965,-44.96986389160156,-40.639095306396484,-36.193790435791016,-49.17943572998047,-44.31909942626953,-45.50566101074219,-46.535255432128906,-30.632104873657227,-49.73768997192383,-25.907108306884766,-41.437313079833984,-47.10711669921875,-19.463016510009766,-33.85370635986328,-40.32176971435547,-45.137413024902344,-42.225833892822266,-42.8878173828125,-15.15355110168457,-29.757001876831055,-35.981651306152344,-12.746479034423828,-41.37387466430664,-35.01076889038086,-40.4298210144043,-30.407438278198242,-28.562698364257812,-47.81247329711914,-41.54508590698242,-20.646913528442383,-47.41183090209961,-41.29351806640625,-47.264305114746094,-49.36017608642578,-44.19395446777344,-47.56047058105469,-37.911136627197266,-39.7394905090332,-42.830360412597656,-34.16742706298828],"y":[10.717097282409668,10.776091575622559,6.513426303863525,8.036958694458008,10.37034797668457,11.778310775756836,11.362831115722656,11.2467622756958,11.140338897705078,8.60769271850586,11.387864112854004,1.2790766954421997,7.539703369140625,3.4808692932128906,9.707032203674316,10.097288131713867,11.729418754577637,9.236603736877441,9.025676727294922,5.413062572479248,10.587983131408691,11.06951904296875,9.410737037658691,9.022080421447754,1.8024154901504517,10.669425010681152,8.293204307556152,10.81798267364502,5.9967942237854,9.815993309020996,3.8676271438598633,2.5082950592041016,6.764708518981934,3.282587766647339,8.280552864074707,10.430875778198242,3.5377092361450195,6.87847375869751,5.8827338218688965,6.579352855682373,2.947702169418335,6.182325839996338,2.738452434539795,9.032819747924805,3.4541096687316895,5.301721096038818,7.121010780334473,5.1406989097595215,10.951601028442383,4.845759868621826,5.750504970550537,1.0027998685836792,-9.027881622314453,0.7447243928909302,9.149815559387207,-3.036175012588501,6.68614387512207,-8.019225120544434,-0.7364959120750427,-4.850863456726074,-2.8502044677734375,-0.8054506182670593,-1.741879940032959,-12.281408309936523,2.906217336654663,-3.9448835849761963,-5.653701305389404,6.257879257202148,8.458844184875488,-8.061084747314453,-9.61425495147705,1.3031928539276123,-1.3214037418365479,2.0234973430633545,1.7442386150360107,-11.903231620788574,1.9529558420181274,11.073802947998047,-8.458504676818848,-4.542888164520264,-3.650989532470703,-1.5122077465057373,5.770267963409424,0.457166463136673,-2.193403959274292,4.535758018493652,0.9542710185050964,-5.215662956237793,-9.4299898147583,-11.27818775177002,-4.607619762420654,-2.9673173427581787,-11.623285293579102,-2.0337157249450684,-12.58556079864502,-13.174449920654297,-6.5448102951049805,-5.661011695861816,-7.609825134277344,-9.930018424987793,-7.419924259185791,0.8691675662994385,0.3845288157463074,-5.879975318908691,-8.375619888305664,-12.374699592590332,0.5747877359390259,0.9893248081207275,-1.2329566478729248,-3.952240228652954,-5.209593772888184,-5.419114112854004,-4.945840835571289,-4.500892162322998,-7.302634239196777,-12.387928009033203,-2.2733283042907715,-10.217785835266113,-10.134686470031738,-2.732412815093994,-3.1192400455474854,-8.403136253356934,-8.141866683959961,-9.349474906921387,-14.539734840393066,-14.508987426757812,-6.953919410705566,-12.59083366394043,-13.519067764282227,-10.17774486541748,-9.25549030303955,-11.983333587646484,-5.440746307373047,-11.595114707946777,-14.031468391418457,-10.57103157043457,-8.959525108337402,-14.724448204040527,-14.897181510925293,-6.588119029998779,-4.374778747558594,-12.01652717590332,-14.207924842834473,-8.888277053833008,-14.551846504211426,-10.70487117767334,-5.7746262550354,-9.057234764099121,-14.656558990478516,-14.663470268249512,-10.069610595703125,-13.019676208496094,-9.880870819091797,-14.68112564086914,-14.633999824523926,-14.616523742675781,-12.861306190490723,-13.355281829833984,-12.714162826538086,-13.861956596374512,-14.615530014038086,-11.342568397521973,-14.086053848266602,-14.635092735290527,-14.64604663848877,-11.214794158935547,-13.239861488342285,-11.805414199829102,-13.896350860595703,-14.562414169311523,-13.62488079071045,-6.289490699768066,-5.985672950744629,-11.2205171585083,-12.692201614379883,-13.548222541809082,-14.136839866638184,-6.258090019226074,-13.97458553314209,-14.121025085449219,-14.196791648864746,-10.533097267150879,-14.657147407531738,-14.607148170471191,-13.923484802246094,-13.626885414123535,-12.07589054107666,-14.511544227600098,-14.34976863861084,-14.660327911376953,-11.42792797088623,-14.378401756286621,-14.369688987731934,-10.52322769165039,-14.222129821777344,-13.825380325317383,-9.225306510925293,-7.644037246704102,-14.158056259155273,-13.423635482788086,-14.686216354370117,-14.475282669067383,16.364683151245117,-14.411798477172852,-14.462105751037598,0.1314530074596405,-14.248242378234863,-3.436068058013916,-13.621509552001953,-11.960503578186035,-11.165571212768555,-5.806784629821777,-14.033788681030273,-10.608258247375488,-13.179341316223145,-5.190380573272705,-14.66439437866211,10.022573471069336,-14.474991798400879,-10.743186950683594,-5.787477493286133,4.856405258178711,-3.1878292560577393,-3.3287508487701416,-14.079508781433105,-11.579689025878906,-13.620491981506348,-8.277702331542969,0.38996800780296326,-5.041080474853516,-14.004733085632324,-14.62918472290039,1.9501742124557495,-14.536812782287598,-7.071580410003662,-4.648726940155029,-13.78942584991455,-10.344719886779785,14.30385684967041,9.270452499389648,-4.231123447418213,15.86470890045166,-5.480589866638184,4.6898674964904785,-10.086644172668457,-6.125968933105469,-11.0060453414917,-10.891369819641113,6.695230007171631,-8.456868171691895,7.564401149749756,-4.417507648468018,-5.518482685089111,-11.810182571411133,5.0593485832214355,-6.437205791473389,15.314608573913574,5.111496448516846,13.500141143798828,0.9384437203407288,-7.497350215911865,-7.017288684844971,6.484698295593262,13.403206825256348,-12.315193176269531,-5.603559970855713,9.751433372497559,8.022343635559082,-7.83704948425293,-8.405556678771973,14.361342430114746,16.201488494873047,-3.7003538608551025,-2.7991490364074707,3.693049192428589,1.6441707611083984,1.5100849866867065,8.147942543029785,15.237377166748047,3.8472466468811035,2.1132729053497314,-1.8707910776138306,12.039670944213867,-6.170930862426758,5.778923988342285,15.591042518615723,10.798894882202148,12.214276313781738,7.456650257110596,-5.580198764801025,-9.756629943847656,3.975703239440918,1.017104148864746,9.220515251159668,14.035898208618164,-7.5305891036987305,-2.0953238010406494,0.42405322194099426,0.44597601890563965,7.309508323669434,14.866287231445312,12.626296043395996,14.854560852050781,5.6608662605285645,11.054272651672363,4.376645088195801,-8.033266067504883,6.751993656158447,-3.1034159660339355,-1.9035983085632324,-10.33910846710205,-4.9859771728515625,10.397544860839844,-10.186209678649902,-9.669608116149902,12.306137084960938,-0.2943928837776184,5.219350814819336,-2.1688921451568604,-8.27153205871582,-2.998894214630127,0.9499883651733398,-5.944852352142334,13.962623596191406,14.433394432067871,15.323562622070312,16.05118751525879,-6.653304100036621,15.986109733581543,6.158205032348633,15.342860221862793,14.63969612121582,15.92805290222168,15.884413719177246,15.400364875793457,4.187515735626221,1.1085432767868042,11.292623519897461,9.53451156616211,9.793724060058594,2.849586009979248,-11.512770652770996,4.228158473968506,9.52463436126709,11.054121971130371,-5.094034671783447,3.221099615097046,3.6058733463287354,-7.569385528564453,15.59573745727539,15.68176555633545,11.429183959960938,10.207715034484863,5.806778430938721,-10.562126159667969,14.638883590698242,16.08292007446289,-3.3143882751464844,1.8832178115844727,-10.110834121704102,10.703845977783203,15.82502555847168,-7.386711120605469,10.194891929626465,13.054861068725586,15.712010383605957,14.925543785095215,-1.8820844888687134,13.229972839355469,15.328715324401855,13.606063842773438,14.279080390930176,-9.128774642944336,9.121521949768066,-9.435653686523438,15.4413480758667,13.327906608581543,14.124800682067871,14.817168235778809,14.900483131408691,14.575288772583008,15.624363899230957,15.440974235534668,14.755621910095215,10.505318641662598,14.954865455627441,4.02458381652832,14.397405624389648,5.0323710441589355,3.4058022499084473,14.919049263000488,15.315402030944824,15.310274124145508,14.772342681884766,4.582414627075195,11.816429138183594,-4.4110870361328125,-3.0016839504241943,14.89875602722168,10.738870620727539,-9.70543098449707,10.331498146057129,4.5591020584106445,6.070935249328613,14.065069198608398,13.359199523925781,0.7829609513282776,6.8668437004089355,5.45451021194458,11.646218299865723,-0.6895207762718201,-2.3479340076446533,2.4240944385528564,-6.1514201164245605,8.899947166442871,10.968256950378418,15.540728569030762,11.545624732971191,4.242351055145264,13.486798286437988,15.586960792541504,7.400242328643799,-8.638232231140137,8.08852767944336,7.6393632888793945,11.229082107543945,-8.877105712890625,8.270744323730469,11.91457748413086,12.816378593444824,15.54238510131836,-3.800934076309204,-7.8186564445495605,2.1381664276123047,0.34422552585601807,12.4141206741333,-9.77180004119873,4.230316638946533,-10.062424659729004,-3.579274892807007,-0.9240943789482117,7.044287204742432,7.496424198150635,6.486103057861328,-0.8481654524803162,4.149562358856201,2.4749386310577393,-5.379260063171387,-1.4789918661117554,4.130218982696533,13.95372486114502,13.261934280395508,12.004082679748535,1.552040934562683,-0.20587149262428284,0.7990784049034119,7.773143291473389,5.163284778594971,6.482840538024902,-9.912313461303711,3.707160472869873,2.41804838180542,-8.268098831176758,13.06359577178955,-7.7627153396606445,-5.964658260345459,-6.9869232177734375,15.673397064208984,-9.949664115905762,-3.237981081008911,13.977652549743652,-0.14131858944892883,4.436889171600342,2.940307855606079,-3.23183536529541,-2.744133234024048,3.072521686553955,13.553244590759277,-2.8042335510253906,16.108232498168945,-9.144908905029297,-10.049642562866211,-2.8726446628570557,11.757792472839355,-8.985451698303223,-9.74885368347168,5.650852680206299,-6.55548095703125,-8.294300079345703,15.25199031829834,5.959709167480469,7.587474346160889,3.271315097808838,11.417298316955566,-9.542637825012207,3.036822557449341,-9.015024185180664,12.299335479736328,7.718315124511719,-7.722607612609863,0.8443247675895691,2.9444847106933594,7.009693145751953,-9.649466514587402,-9.888447761535645,5.815920829772949,13.772531509399414,-9.705497741699219,12.118334770202637,2.0229380130767822,4.890763282775879,-3.045154094696045,15.20584774017334,-6.285811424255371,-9.1874361038208,13.411462783813477,-9.069089889526367,4.732192516326904,-1.677880048751831,-3.7545204162597656,-10.145370483398438,8.340934753417969,-2.284599542617798,0.010624407790601254,2.712035655975342,3.308253049850464,7.903101444244385,-9.051817893981934,8.462508201599121,-9.84355354309082,0.9125211238861084,-1.8285762071609497,-8.93624496459961,-0.39259839057922363,-9.80745792388916,2.628418207168579,13.13290786743164,5.415491104125977,-9.95596981048584,5.774054527282715,15.656709671020508,11.163379669189453,-8.761666297912598,-8.288928985595703,-4.781616687774658,4.680839538574219,6.366890907287598,-9.788269996643066,14.130121231079102,-0.19151873886585236,-7.138979434967041,-6.828821182250977,12.51927375793457,-8.3800048828125,-9.570242881774902,-3.683211088180542,-7.549582481384277,0.8365201950073242,6.193172931671143,-8.363373756408691,6.912486553192139,-6.506660461425781,-5.092291831970215,5.810415267944336,-9.185749053955078,3.1589322090148926,5.421520233154297,1.1846202611923218,-5.466179370880127,-0.1161857321858406,1.096400499343872,11.472415924072266,-1.1199123859405518,-9.614068031311035,3.2594592571258545,-8.168736457824707,-5.381218910217285,-5.778344631195068,-8.511855125427246,1.25178062915802,5.330306053161621,-6.822178363800049,-8.442399024963379,-1.059792399406433,-8.890732765197754,-7.020555019378662,-9.458162307739258,-10.615375518798828,7.060028553009033,15.314909934997559,-9.659270286560059,1.0474472045898438,-5.13385009765625,12.027724266052246,-7.65725564956665,-9.248627662658691,-9.249406814575195,3.1440672874450684,3.9227488040924072,-9.70908260345459,-0.5325383543968201,-10.389877319335938,-6.504161357879639,1.5900071859359741,7.546756267547607,-6.783273696899414,-5.970954418182373,5.134756565093994,2.3529365062713623,6.7778425216674805,4.226531505584717,-8.385915756225586,14.743538856506348,0.3737327754497528,-2.5154149532318115,-3.744169235229492,-9.393369674682617,4.787927150726318,-6.232975006103516,3.2268900871276855,0.645940899848938,-10.128072738647461,-5.733193874359131,7.514043807983398,-2.960148572921753,-9.872356414794922,-8.503619194030762,4.409868240356445,-9.540050506591797,-3.192147970199585,4.151045799255371,0.4836297929286957,-8.064444541931152,12.901238441467285,-7.935359477996826,-1.845445990562439,-10.522034645080566,-2.429002046585083,-2.0046327114105225,3.8309383392333984,-0.571754515171051,-7.647136688232422,-8.360739707946777,-9.638517379760742,7.0527753829956055,-6.186689376831055,7.464146137237549,7.3585205078125,-9.443405151367188,-7.22700309753418,12.708887100219727,-6.840198516845703,0.6900372505187988,-4.353083610534668,-1.3391708135604858,-10.510894775390625,0.09437716752290726,-8.899398803710938,3.5583269596099854,-8.752630233764648,-2.6043388843536377,1.953413486480713,-7.047938823699951,-9.904869079589844,1.8395559787750244,-3.916249990463257,-7.457711219787598,7.038022994995117,1.6398017406463623,2.598339080810547,3.9624221324920654,-9.981694221496582,7.908668041229248,-9.740584373474121,-2.4489314556121826,4.039005756378174,-6.015420436859131,-9.554546356201172,-4.160758972167969,2.0227770805358887,-1.7475370168685913,-0.7408920526504517,0.8347991704940796,-10.561358451843262,-7.620830059051514,6.773906230926514,-2.7274465560913086,-8.578822135925293,-3.3750159740448,-10.292433738708496,-10.33691120147705,4.70911169052124,-2.5715348720550537,-7.262990951538086,4.980209827423096,-3.639204740524292,4.097658157348633,7.266537189483643,1.012233018875122,4.99732780456543,-6.321319580078125,-4.6139655113220215,-0.7677422165870667,-9.155485153198242],"type":"scatter"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('dbdb2535-5dad-4a76-bd4a-368fbc6f57cf');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
```


```python
#establish an empty dictionary
embeddings_dict = {}

#open the file and read it into the dictionary
with open("glove.6B/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[25], line 5
          2 embeddings_dict = {}
          4 #open the file and read it into the dictionary
    ----> 5 with open("glove.6B/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
          6     for line in f:
          7         values = line.split()


    File ~/anaconda3/lib/python3.10/site-packages/IPython/core/interactiveshell.py:282, in _modified_open(file, *args, **kwargs)
        275 if file in {0, 1, 2}:
        276     raise ValueError(
        277         f"IPython won't let you open fd={file} by default "
        278         "as it is likely to crash IPython. If you know what you are doing, "
        279         "you can use builtins' open."
        280     )
    --> 282 return io_open(file, *args, **kwargs)


    FileNotFoundError: [Errno 2] No such file or directory: 'glove.6B/glove.6B.100d.txt'



```python
#establish an empty dictionary
embeddings_dict = {}

#open the file and read it into the dictionary
with open("glove.6B.100d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
```


```python
#find the Euclidean distance between the vectors for words and 1 or more other words.
#sort the resulting word distances.
def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: 
                  spatial.distance.euclidean(embeddings_dict[word], embedding))
```


```python
print(find_closest_embeddings(
    embeddings_dict["dog"]
)[:20])
```

    ['dog', 'cat', 'dogs', 'puppy', 'pet', 'horse', 'pig', 'cats', 'animal', 'rabbit', 'boy', 'goat', 'monkey', 'rat', 'hound', 'breed', 'canine', 'sled', 'pets', 'puppies']



```python
print(find_closest_embeddings(
    embeddings_dict["cat"]
)[:20])
```

    ['cat', 'dog', 'rabbit', 'cats', 'monkey', 'puppy', 'pet', 'dogs', 'rat', 'mouse', 'spider', 'elephant', 'ghost', 'pig', 'monster', 'ape', 'parrot', 'squirrel', 'kitten', 'hound']



```python
print(find_closest_embeddings(
    embeddings_dict["girl"]
)[:20])
```

    ['girl', 'boy', 'woman', 'girls', 'mother', 'teenager', 'girlfriend', 'child', 'teenage', 'teen', 'boys', 'kid', 'mom', 'baby', 'man', 'couple', 'sister', 'boyfriend', 'toddler', 'ager']



```python
print(find_closest_embeddings(
    embeddings_dict["house"]
)[:20])
```

    ['house', 'office', 'room', 'capitol', 'houses', 'mansion', 'once', 'home', 'senate', 'building', 'door', 'came', 'where', 'congressional', 'clinton', 'turned', 'floor', 'hill', 'now', 'next']



```python
print(find_closest_embeddings(
    embeddings_dict["bike"]
)[:20])
```

    ['bike', 'bicycle', 'bikes', 'ride', 'rides', 'riding', 'motorcycle', 'biking', 'bicycles', 'horseback', 'rode', 'riders', 'walking', 'snowmobile', 'wagon', 'cart', 'skateboard', 'bicycling', 'motorbike', 'driving']



```python
print(find_closest_embeddings(
    embeddings_dict["pool"]
)[:20])
```

    ['pool', 'pools', 'swimming', 'room', 'table', 'playground', 'outdoor', 'tub', 'stands', 'tables', 'addition', 'filled', 'goals_none', 'jacuzzi', 'floor', 'fill', 'setting', 'nasdaq100', 'splash', 'placed']



```python
print(find_closest_embeddings(
    embeddings_dict["sad"]
)[:20])
```

    ['sad', 'sorry', 'awful', 'tragic', 'horrible', 'heartbreaking', 'unfortunate', 'pathetic', 'scary', 'happy', 'poignant', 'shocking', 'sorrowful', 'terrible', 'ugly', 'sadly', 'lonely', 'unhappy', 'confused', 'depressing']



```python
print(find_closest_embeddings(
    embeddings_dict["art"]
)[:20])
```

    ['art', 'arts', 'museum', 'sculpture', 'works', 'photography', 'contemporary', 'painting', 'gallery', 'collection', 'architecture', 'exhibit', 'exhibition', 'artist', 'collections', 'culture', 'architectural', 'artwork', 'paintings', 'artistic']



```python
print(find_closest_embeddings(
    embeddings_dict["book"]
)[:20])
```

    ['book', 'books', 'novel', 'wrote', 'essay', 'author', 'biography', 'story', 'published', 'memoir', 'titled', 'written', 'autobiography', 'writing', 'publication', 'illustrated', 'describes', 'fiction', 'read', 'novels']



```python
print(find_closest_embeddings(
    embeddings_dict["computer"]
)[:20])
```

    ['computer', 'computers', 'software', 'technology', 'hardware', 'pc', 'computing', 'electronic', 'laptop', 'desktop', 'internet', 'systems', 'web', 'applications', 'user', 'digital', 'devices', 'ibm', 'multimedia', 'virtual']



```python
print(find_closest_embeddings(
    embeddings_dict["code"]
)[:20])
```

    ['code', 'codes', 'rules', 'defines', 'instance', 'qnix', 'system', 'example', 'standard', 'specifies', 'regulations', 'defined', 'standards', 'allows', 'file', 'refers', 'introduced', 'reference', 'uses', 'requires']



```python
print(find_closest_embeddings(
    embeddings_dict["rose"]
)[:20])
```

    ['rose', 'fell', 'climbed', 'surged', 'dropped', 'jumped', 'slipped', 'soared', 'tumbled', 'dipped', 'gained', 'risen', 'shares', 'falling', 'slid', 'slumped', 'declined', 'rise', 'plummeted', 'plunged']



```python
print(find_closest_embeddings(
    embeddings_dict["flower"]
)[:20])
```

    ['flower', 'flowers', 'leaf', 'floral', 'roses', 'petals', 'garden', 'lavender', 'fruit', 'bloom', 'buds', 'jasmine', 'blossoms', 'purple', 'lily', 'nursery', 'blossom', 'tree', 'shade', 'clover']



```python
print(find_closest_embeddings(
    embeddings_dict["dog"] + embeddings_dict["cat"]
)[:20])
```

    ['dog', 'cat', 'dogs', 'pet', 'rabbit', 'horse', 'puppy', 'cats', 'monkey', 'animal', 'mouse', 'boy', 'pig', 'rat', 'bird', 'baby', 'duck', 'snake', 'cow', 'goat']



```python
print(find_closest_embeddings(
    embeddings_dict["flower"] + embeddings_dict["water"]
)[:20])
```

    ['water', 'flower', 'flowers', 'red', 'natural', 'fruit', 'green', 'garden', 'large', 'dry', 'tree', 'light', 'food', 'small', 'fields', 'fish', 'tea', 'trees', 'yellow', 'sand']



```python
print(find_closest_embeddings(
    embeddings_dict["school"] + embeddings_dict["cat"]
)[:20])
```

    ['school', 'college', 'boys', 'boy', 'girls', 'where', 'girl', 'cat', 'dog', 'kids', 'she', 'student', 'home', 'teacher', 'children', 'schools', 'child', 'high', 'one', 'little']



```python
print(find_closest_embeddings(
    embeddings_dict["dog"] + embeddings_dict["cat"] + embeddings_dict["pet"]
)[:20])
```

    ['dog', 'pet', 'cat', 'dogs', 'animal', 'cats', 'pets', 'animals', 'puppy', 'baby', 'rabbit', 'mouse', 'cow', 'monkey', 'bird', 'horse', 'pig', 'rat', 'boy', 'duck']



```python
words =  list(embeddings_dict.keys())
vectors = [embeddings_dict[word] for word in words]
X = np.asarray(vectors)
```


```python
tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(X[:1000])
```


```python
plt.scatter(Y[:, 0], Y[:, 1])
```




    <matplotlib.collections.PathCollection at 0x2b89d0910>




    
![png](output_43_1.png)
    



```python
for label, x, y in zip(words, X[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()
```


    
![png](output_44_0.png)
    



```python

```
