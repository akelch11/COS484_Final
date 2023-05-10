import re
from bert_utils import *
from transformers import InputExample
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')


class Interpreter():
    '''Abstract class for all interpreters'''
    def __init__(self):
        return

    def batch_interpret(self, inputs, explanation_list):
        '''
        The interpret functionality but for batched input.
        Works faster if the interpreter involves a neural net as the
        interpreter
        '''
        return

    def interpret(self, input_x, explanation):
        '''
        Interprets the explanation in the context of the input
        producing a feature grounding the explanation in the input
        '''
        return


class RegexInterpreter(Interpreter):
    def __init__(self, k=3):
        super(RegexInterpreter, self).__init__()
        self.ngrams = k

    def convert_to_ngrams(self, explanation):
        all_words = explanation.split(" ")
        all_n_grams = []
        num_words = len(all_words)
        for k in range(1, self.ngrams+1):
            all_n_grams += [" ".join(all_words[i : i+k]) for\
                    i in range(num_words - k)]
        return all_n_grams

    def compile_to_regex(self, all_n_grams):
        regexes = []
        for n_gram in all_n_grams:
            regex = re.compile(n_gram.lower())
            regexes.append(regex)
        return regexes

    def batch_interpret(self, inputs, explanation_list):
        all_n_grams = []
        for explanation in explanation_list:
            all_n_grams += self.convert_to_ngrams(explanation)
        all_n_grams = list(set(all_n_grams))
        # ensuring that we always have a consistent order
        all_n_grams.sort()
        regexes = self.compile_to_regex(all_n_grams)
        print(all_n_grams)
        regex_features = []
        for input_x, _ in inputs:
            curr_features = self._interpret(input_x, regexes)
            regex_features.append(curr_features)
        regex_features = np.array(regex_features)
        return regex_features

    def _interpret(self, input_x, regex_list):
        input_text, e_1, e_2 = input_x
        text = input_text.lower()
        feats = [int(pat.search(text) != None) for pat in regex_list]
        return feats

    def interpret(self, input_x, explanation):
        '''
        Converts the explanation into a collection of n-grams with n = 1,2,..k
        and returns a binary feature indicating whether the n-gram matches against
        the input or not
        '''
        all_n_grams = self.convert_to_ngrams(explanation)
        regexes = self.compile_to_regex(all_n_grams)
        return self._interpret(input_x, regexes)



# TODO: the inputs need to have the 2 entities as well...
class BertInterpreter(Interpreter):
    def __init__(self, bert_model, tokenizer, device, use_logits=False, max_seq_len=128):
        super(BertInterpreter, self).__init__()
        bert_model.to(device)
        self.model = bert_model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

        self.device = device
        self.use_logits = use_logits

    def process(self, input_x, explanation, idx):
        input_text, e_1, e_2 = input_x
        explanation = explanation.replace('{e1}', f'{e_1}').replace('{e2}', f'{e_2}')
        example =  InputExample(guid=idx, text_a=input_text, text_b=explanation)
        return example

    def batch_interpret(self, inputs, explanation_list):
        examples = []
        idx = 0
        print(explanation_list)
        print(len(explanation_list))
        for input_x, _ in inputs:
            for explanation in explanation_list:
                examples.append(self.process(input_x, explanation, idx))
                idx += 1

        # required for huggingface
        features = convert_examples_to_features(examples,
                                                self.tokenizer,
                                                max_length=self.max_seq_len,
                                                pad_token=self.pad_token
                                                )
        #TODO: possibly cache this for using later
        tensors = convert_to_tensors(features)
        features = run_bert(self.model,
                            self.tokenizer,
                            tensors,
                            use_logits=self.use_logits,
                            device=self.device)
        return features.reshape(len(inputs), len(explanation_list), -1)

    def interpret(self, input_x, explanation):
        # typically slow for BERT - better to use the batched version
        pass


class ElectraInterpreter(Interpreter):
    def __init__(self, electra_model, tokenizer, device, use_logits=False, max_seq_len=128):
        super(ElectraInterpreter, self).__init__()
        electra_model.to(device)
        self.model = electra_model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

        self.device = device
        self.use_logits = use_logits

    def process(self, input_x, explanation, idx):
        input_text, e_1, e_2 = input_x
        explanation = explanation.replace('{e1}', f'{e_1}').replace('{e2}', f'{e_2}')
        example =  InputExample(guid=idx, text_a=input_text, text_b=explanation)
        return example

    def batch_interpret(self, inputs, explanation_list):
        examples = []
        idx = 0
        print(explanation_list)
        print(len(explanation_list))
        for input_x, _ in inputs:
            for explanation in explanation_list:
                examples.append(self.process(input_x, explanation, idx))
                idx += 1

        # required for huggingface
        features = convert_examples_to_features(examples,
                                                self.tokenizer,
                                                max_length=self.max_seq_len,
                                                pad_token=self.pad_token
                                                )
        #TODO: possibly cache this for using later
        tensors = convert_to_tensors(features)
        features = run_electra(self.model,
                            self.tokenizer,
                            tensors,
                            use_logits=self.use_logits,
                            device=self.device)
        return features.reshape(len(inputs), len(explanation_list), -1)

class RobertaInterpreter(Interpreter):
    def __init__(self, roberta_model, tokenizer, device, use_logits=False, max_seq_len=128):
        super(RobertaInterpreter, self).__init__()
        roberta_model.to(device)
        self.model = roberta_model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

        self.device = device
        self.use_logits = use_logits

    def process(self, input_x, explanation, idx):
        input_text, e_1, e_2 = input_x
        explanation = explanation.replace('{e1}', f'{e_1}').replace('{e2}', f'{e_2}')
        example =  InputExample(guid=idx, text_a=input_text, text_b=explanation)
        return example

    def batch_interpret(self, inputs, explanation_list):
        examples = []
        idx = 0
        print(explanation_list)
        print(len(explanation_list))
        for input_x, _ in inputs:
            for explanation in explanation_list:
                examples.append(self.process(input_x, explanation, idx))
                idx += 1

        # required for huggingface
        features = convert_examples_to_features_roberta(examples,
                                                self.tokenizer,
                                                max_length=self.max_seq_len,
                                                pad_token=self.pad_token
                                                )
        #TODO: possibly cache this for using later
        tensors = convert_to_tensors_roberta(features)
        features = run_roberta(self.model,
                            self.tokenizer,
                            tensors,
                            use_logits=self.use_logits,
                            device=self.device)
        return features.reshape(len(inputs), len(explanation_list), -1)


class WordToVecInterpreter(Interpreter):
    def __init__(self):
        super(WordToVecInterpreter, self).__init__()
        return

    def process(self, input_x, explanation, idx):
        input_text, e_1, e_2 = input_x
        explanation = explanation.replace('{e1}', f'{e_1}').replace('{e2}', f'{e_2}')
        example = input_text + " " + explanation
        return example
    


    def batch_interpret(self, inputs, explanation_list):
        examples = []
        idx = 0
        print(explanation_list)
        print(len(explanation_list))
        for input_x, _ in tqdm(inputs):
            for explanation in explanation_list:
                examples.append(self.process(input_x, explanation, idx))
                idx += 1

        ft = fasttext.load_model('cc.en.300.bin')
        feats = np.zeros((len(inputs), len(explanation_list) ,300))
        print(feats.shape)

        i = 0
        j = 0



        for example in tqdm(examples):
            if (i == len(explanation_list) and i != 0):
                j = j + 1
                i = 0

            words = word_tokenize(example)
            final_vec = np.zeros(300)

            for word in words:
                vec = ft.get_word_vector(word)
                final_vec = vec + final_vec

            mean_vec = final_vec / len(words)
            feats[j][i] = mean_vec
            i = i + 1

        return feats

