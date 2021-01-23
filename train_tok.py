import logging
from pathlib import Path
from tokenizers import BertWordPieceTokenizer



def train_tokenizer(data_file_paths, vocab_size):
    t = BertWordPieceTokenizer(handle_chinese_chars=False)
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    wordpieces_prefix = "##"
    t.train(
        files=data_file_paths,
        vocab_size=vocab_size,
        min_frequency=10,
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=1000,
        wordpieces_prefix=wordpieces_prefix,

    )
    return t


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# vocab_size = 52000
vocab_size = 2000
# paths = [str(x) for x in Path("data/raw").glob("oscar/he_dedup≥.txt")]
paths = [r"C:\Dev\Models_Ex3\dataset\develop_no_preprocess.txt"]
tokenizer = train_tokenizer(paths, vocab_size)
tokenizer.save_model(f'./experiments/tokenizers/wp-{vocab_size}')
