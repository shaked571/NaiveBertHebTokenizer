from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers.utils import logging
from typing import List
import collections
import os

logger = logging.get_logger(__name__)

class AlefBERTNaiveTokenizer(BertTokenizer):
    r"""
    Construct a AlefBERTNaiveTokenizer  tokenizer. Based on WordPiece (and Bert Tokenizer).

    This tokenizer inherits from :class:`~transformers.models.bert.tokenization_bert.BertTokenizer` which contains most
    of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    rule_file = 'prefix.utf8'
    rule_path = os.path.abspath(os.path.join(os.path.dirname(__file__),  rule_file))

    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=False,
            tokenize_chinese_chars=False
            **kwargs
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'".format(vocab_file)
            )
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self._NOT_TO_SPlIT = {'של', 'שלכם', 'שלנו', 'שלהם', 'שלך', 'שלי', 'מי', 'מה'}
        self.rules = self.get_prefix_rules()
        self.prefix_rules = dict(self.rules)
        self.tokenize_chinese_chars=False


    def get_prefix_rules(self, path=rule_path)->List[List[str]]:
            rules = []
            with open(path, mode="r", encoding='utf-8') as f:
                for l in f:
                    rule = l.split()
                    if rule is not None:
                        rules.append(rule)
            rules = sorted(rules, key=lambda x: len(x[0]), reverse=True)
            return rules


    def get_longest_prefix(self, t):
        for r in self.rules:
            if t.startswith(r[0]):  # rules are sorted from the longest to shortest
                return r[0]
        return None

    def break_word(self, word, rule):
        sub_t = rule.split('^')
        suffix = word.split("".join(sub_t), 1)[1]
        res = " " + " ".join(sub_t) + " " + suffix
        return res

    def pre_tok(self, text: str) -> str:
        res = ''
        txt_split = text.split()
        for t in txt_split:
            if any([t.startswith(c) for c in self.prefix_rules]) and t not in self._NOT_TO_SPlIT:
                lp = self.get_longest_prefix(t)
                if lp is None or len(t) < len(lp) + 2:
                    res += f" {t}"
                    continue
                rule = self.prefix_rules[lp]
                res += self.break_word(t, rule)
            else:
                res += f" {t}"
        return res[1:]  # remove the first redundant space

    def _tokenize(self, text):
        split_tokens = []
        text = self.pre_tok(text)
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens
