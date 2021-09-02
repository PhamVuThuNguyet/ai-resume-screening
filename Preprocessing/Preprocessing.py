import string
import re
import nltk
from nltk.tokenize import TweetTokenizer
import unidecode
import os


def get_all_txt_paths(directory):
    """
    get all of pdf file path in directory
    :param directory: directory to get file
    :type directory: string
    :return: list of file paths
    :rtype: list
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_paths.append(os.path.join(root, file))
    return file_paths


def preprocessing(path_list, save_dir):
    for p in path_list:
        path, filename = os.path.split(p)

        fp = open(p, 'r', encoding='utf8')

        text = fp.read()
        text_no_spe_char = remove_special_char(text)
        text_no_extra_space = remove_extra_space(text_no_spe_char)
        text_lower = lower_case(text_no_extra_space)
        text_no_incoherent_punct = remove_punctuation_first(text_lower)
        token = tokenize(text_no_incoherent_punct)
        # text_full_eng = process_non_eng_char(token)
        text_no_unicode = remove_unicode_character(token)
        text_no_stop_words = remove_stopwords(text_no_unicode)
        text_no_punct = remove_punctuation(text_no_stop_words)
        new_text = ' '.join(text_no_punct)

        file_to_save = save_dir + filename

        f = open(file_to_save, "w", encoding="utf-8")
        f.write(new_text)
        print("Clean", file_to_save)
        fp.close()
        f.close()


# remove special symbol
def remove_special_char(text):
    pattern = r'[➢•▪“”’·©●●•®❖*' \
              r'❒*>✓❏]'
    mod_string = re.sub(pattern, '', text)
    return mod_string


# remove redundant whitespaces
def remove_extra_space(text):
    text_without_redundant_space = re.sub("\s+", " ", text)
    return text_without_redundant_space


# lower case
def lower_case(text):
    lower = [words.lower() for words in text]
    text_lower = ''.join(lower)
    return text_lower


# remove punctuation
def remove_punctuation_first(text):
    regex = r"(?<!\d)[.,;:](?!\d)"
    gruber = re.compile(r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|
biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|
az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|
de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|
hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|
ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|
ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|
td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)
(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|
[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|
coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|
bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|
dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|
ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|
md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|
pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|
th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""")

    final_text = "".join(t if i % 2 else t.translate(regex) for (i, t) in enumerate(gruber.split(text)))

    return final_text


# tokenize
def tokenize(text):
    tweet_token = TweetTokenizer()
    return tweet_token.tokenize(text)


# remove punctuation
def remove_punctuation(text):
    text_without_punctuation = [w for w in text if not (w in string.punctuation or w == '...' or w == '..' or w == '…')
                                or (w == "+") or (w == "#")]
    return text_without_punctuation


# remove stop words
def remove_stopwords(text):
    stop_words = nltk.corpus.stopwords.words('english')
    text_without_stop_words = [word for word in text if word not in stop_words]
    return text_without_stop_words


# make non-english char to english char
def process_non_eng_char(text):
    text_without_non_eng = [unidecode.unidecode(word) for word in text]
    return text_without_non_eng


# remove unicode character:
def remove_unicode_character(text):
    decode = [words.encode('ascii', 'ignore').decode() for words in text]
    return decode


# stemming & lemma
def stemming_lemma(text):
    wn = nltk.WordNetLemmatizer()
    w = [wn.lemmatize(word) for word in text]
    return w


# remove single letter
def remove_single_letter(text):
    text_without_single_letter = re.sub(r"\b[a-zA-Z]\b", "", text)
    return text_without_single_letter


if __name__ == '__main__':
    dir_ = "../data/converted_txt_JD"
    path_list_ = get_all_txt_paths(dir_)
    save_dir_ = "../data/preprocessed_txt_JD/"
    preprocessing(path_list_, save_dir_)


