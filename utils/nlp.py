import numpy
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from tqdm import tqdm
from multiprocessing import Pool


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return text.split()



def preprocess_(dataset):
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time',
                   'date', 'number'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis',
                  'censored'},
        all_caps_tag="wrap",
        fix_text=True,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    ).pre_process_doc
    return [preprocessor(x) for x in dataset]

def twitter_preprocess():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time',
                   'date', 'number'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis',
                  'censored'},
        all_caps_tag="wrap",
        fix_text=True,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    ).pre_process_doc

    def preprocess(name, dataset):
        desc = "PreProcessing dataset {}...".format(name)

        data = [None for _ in range(len(dataset))]
        N = len(data)
        for i, x in tqdm(enumerate(dataset), desc=desc, total=N):
            data[i] = preprocessor(x)
        data = [preprocessor(x)
                for x in tqdm(dataset, desc=desc)]

        return data

    def parallel_preprocess(name, dataset):
        N = len(dataset)
        batchsize = 1000
        n_splits = N // batchsize + (1 if N % batchsize > 0 else 0)
        batches = (dataset[i*batchsize:(i+1)*batchsize] for i in range(n_splits))
        data = []
        with Pool(processes=6) as p:
            for result in tqdm(p.imap(preprocess_, batches), total=n_splits):
                data += result
        return data

    # return preprocess
    return parallel_preprocess


def vectorize(sequence, el2idx, max_length, unk_policy="random",
              spell_corrector=None):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        sequence (): a list of elements
        el2idx (): dictionary of word to ids
        max_length ():
        unk_policy (): how to handle OOV words
        spell_corrector (): if unk_policy = 'correct' then pass a callable
            which will try to apply spell correction to the OOV token


    Returns: list of ids with zero padding at the end

    """
    words = numpy.zeros(max_length).astype(int)

    # trim tokens after max length
    sequence = sequence[:max_length]

    for i, token in enumerate(sequence):
        if token in el2idx:
            words[i] = el2idx[token]
        else:
            if unk_policy == "random":
                words[i] = el2idx["<unk>"]
            elif unk_policy == "zero":
                words[i] = 0
            elif unk_policy == "correct":
                corrected = spell_corrector(token)
                if corrected in el2idx:
                    words[i] = el2idx[corrected]
                else:
                    words[i] = el2idx["<unk>"]

    return words
