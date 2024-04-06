import re
import unicodedata
import zipfile
import math

zip = 'DataFP (1).zip'
### FROM: https://stackoverflow.com/a/518232/3025981
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def clean_text(s):
    return re.sub("[^a-z \n]", "", strip_accents(s))


def get_freqs(text, relative=False):
    lang_to_prob = dict()
    for el in text:
        if el not in lang_to_prob:
            lang_to_prob[el] = 1
        else:
            lang_to_prob[el] += 1
    return lang_to_prob


def probs_dict():
    with zipfile.ZipFile(zip, 'r') as f:
        lang_to_prob = dict()
        file_names = f.namelist()
        not_decodable = 0
        counter = 0
        for file_name in file_names:
            counter = 0
            file_name_cleaned = file_name.rstrip('.txt')
            lang_to_prob[file_name_cleaned] = dict()
            with f.open(file_name, 'r') as inner_file:
                read_lines = inner_file.readlines()
                decoded_lines = [line.decode("utf-8") for line in read_lines]
                cleaned_lines = [clean_text(line) for line in decoded_lines]
                for line in cleaned_lines:
                    for char in line:
                        counter += 1
                        lang_to_prob[file_name_cleaned][char] = lang_to_prob[file_name_cleaned].get(char, 0) + 1
                for el in lang_to_prob[file_name_cleaned]:
                    lang_to_prob[file_name_cleaned][el] /= counter

    return lang_to_prob

#print(probs_dict())

def multinomial_likelihood(probs, freqs):
    multinomial_coefficient = math.factorial(sum(freqs.values()))
    for freq in freqs.values():
        multinomial_coefficient //= math.factorial(freq)

    #probabilities
    for char in freqs:
        multinomial_coefficient *= (probs[char] ** freqs[char])
    return multinomial_coefficient


def multinomial_likelihood_without_coeff(probs, freqs):
    multinomial_coefficient = 1
    for char in freqs:
        multinomial_coefficient *= (probs[char] ** freqs[char])
    return multinomial_coefficient
res = multinomial_likelihood_without_coeff(probs={'a': 0.2, 'b': 0.5, 'c': 0.3},
                                     freqs={'a': 2, 'b': 1, 'c': 2})

def log_likelihood_without_coeff(probs, freqs):
    log_likelihood = 0
    for char, freq in freqs.items():
        if char in probs:
            log_likelihood += freq * math.log(probs[char])
    return log_likelihood

lang_to_probs = probs_dict()
# for el in lang_to_probs:
#     print(lang_to_probs[el])
def mle_best(text, lang_to_probs):
    cleaned_text = clean_text(text)
    max_likelihood = -math.inf
    max_likelihood_key = str()
    chars_freqs = dict()
    likelihood_dict = dict()
    for char in text:
        chars_freqs[char] = chars_freqs.get(char, 0) + 1
    for dict_of_probs in lang_to_probs:
        likelihood = log_likelihood_without_coeff(lang_to_probs[dict_of_probs],chars_freqs)
        likelihood_dict[dict_of_probs] = likelihood
    for el in likelihood_dict:
        if likelihood_dict[el] > max_likelihood:
            max_likelihood_key = el
            max_likelihood = likelihood_dict[el]
    return max_likelihood_key

lang_to_prior = {'English': 6090, 'Italian': 1611, 'Spanish': 1602, 'German': 2439, 'French': 2222, 'Polish': 1412, 'Portuguese': 1034}
sum = sum(lang_to_prior.values())
lang_to_prior = {lang: value / sum for lang, value in lang_to_prior.items()}

def bayesian_best(text, lang_to_probs, lang_to_prior):
    cleaned_text = clean_text(text)
    text_freqs = get_freqs(cleaned_text)
    best_lang = None
    max_posterior = -math.inf
    for lang, probs in lang_to_probs.items():
        if lang in lang_to_prior:
            log_posterior = log_likelihood_without_coeff(probs, text_freqs) + math.log(lang_to_prior[lang])
            if log_posterior > max_posterior:
                max_posterior = log_posterior
                best_lang = lang
    return best_lang


