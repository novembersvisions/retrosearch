
import re
import math

def tokenize(text: str) -> list[str]:
    """Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything. Do not remove duplicate words.
    Requirement: Use Regex to satisfy this function
    
    Parameters
    ----------
    text : str
        The input string to be tokenized.

    Returns
    -------
    List[str]
        A list of strings representing the words in the text.
    """
    words = re.findall('[A-Za-z]+', text.lower())
    return words

def build_inverted_index(msgs: list[dict]) -> dict:
    """Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (title, count_of_term_in_doc)
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]

    """
    term_index = {}
    count = 0
    for d in msgs:
      word_dict = {}
      title = count
      for word in d['toks']:
        word_dict[word] = word_dict.get(word, 0) + 1
      for w in word_dict:
        if w not in term_index:
          term_index[w] = [(title,word_dict[w])]
        else: 
          term_index[w].append((title,word_dict[w]))
      count += 1
    return dict(sorted(term_index.items(), key=lambda x: x[0]))

def tf(t,text):
  """ Calculates term frequency of term ``t`` in the abstract of JSON string ``text`` """
  abstr = text['abstract']
  return abstr.count(t) / len(abstr)

def idf(t,inv_index,total):
   """ Calculates inverse document frequency of term ``t`` given the inverted index ``inv_index`` and total number of texts ``total``. """
   return math.log(total / len(inv_index[t]))
