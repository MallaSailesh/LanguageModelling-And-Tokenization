import sys, pickle, os
from tokenizer import MyTokenizer
from n_gram import generate_ngram_model
from language_model import LM_GoodTuring, LM_Interpolation

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_word_before_smoothing(k, path, txt, N=3):
    k = int(k)
    with open(path, 'r', encoding='utf-8') as f:
        train_txt = f.read()

    tokenizer = MyTokenizer()
    train_txt = tokenizer.replace_tokens_with_placeholders(train_txt)
    train_tokens = tokenizer.tokenize(train_txt)

    txt = tokenizer.replace_tokens_with_placeholders(txt)
    tokens = tokenizer.tokenize_words(txt) # Assuming given a sentence 
    tokens = ["<start>"]*(N-1) + tokens 
    num_tokens = len(tokens)
    tokens = tuple(tokens[num_tokens-N+1:]) # based on this we need to predict the next word

    lm = generate_ngram_model(train_tokens, N)

    if tokens in lm.keys():
        sorted_elements = sorted(lm[tokens].items(), key = lambda x: x[1], reverse=True)
        if len(sorted_elements) == 1:
            print(f"\033[1mNext Predicted words are less than {k}.\033[0m")
        elif len(sorted_elements)-1 < k:
            total = lm[tokens]["TotalCnt"]
            for pair in sorted_elements:
                if pair[0] != "TotalCnt":
                    print(f"{pair[0]} {pair[1]/total}")
            print(f"\033[1mNext Predicted words are less than {k}.\033[0m")
        else:
            total = lm[tokens]["TotalCnt"]
            left = k
            for pair in sorted_elements:
                if left == 0:
                    break
                if pair[0] != "TotalCnt":
                    print(f"{pair[0]} {pair[1]/total}")
                    left-=1
    else:
        print(f"\033[1mNext Predicted words are less than {k}.\033[0m")
    
    print("\n\033[1mOf Course higher the value of n, higher the fluency because it clearly predicts the words based on history of bigger size.\033[0m")

def predict_word_after_smoothing(k, path, txt, smoothing_type, N=3):
    k = int(k)
    with open(path, 'r', encoding='utf-8') as f:
        train_txt = f.read()

    tokenizer = MyTokenizer()
    train_txt = tokenizer.replace_tokens_with_placeholders(train_txt)
    train_tokens = tokenizer.tokenize_sentences(train_txt)

    txt = tokenizer.replace_tokens_with_placeholders(txt)
    tokens = tokenizer.tokenize_words(txt) # Assuming given a sentence 
    tokens = ["<start>"]*(N-1) + tokens 
    num_tokens = len(tokens)
    tokens = tuple(tokens[num_tokens-N+1:]) # based on this we need to predict the next word

    lm_idx = 1
    if smoothing_type == 'g' and path == './corpus/2.txt':
        lm_idx = 3
    if smoothing_type == 'i' and path == './corpus/1.txt':
        lm_idx = 2
    if smoothing_type == 'i' and path == './corpus/2.txt':
        lm_idx = 4

    if smoothing_type == 'g':
        final_prob = load_model(f"./models/LM{lm_idx}_model.pkl")
        if tokens in final_prob.keys():
            sorted_elements = sorted(final_prob[tokens].items(), key = lambda x: x[1], reverse=True)
            if len(sorted_elements)-1 < k:
                total = final_prob[tokens]["TotalProb"]
                for pair in sorted_elements:
                    if pair[0] != "TotalProb":
                        print(f"{pair[0]} {pair[1]/total}")
                print(f"<unk> {final_prob['<unk>']}")
                print(f"\033[1mNext Predicted words are less than {k}.\033[0m")
            else:
                total = final_prob[tokens]["TotalProb"]
                left = k
                for pair in sorted_elements:
                    if left == 0:
                        break
                    if pair[0] != "TotalProb":
                        print(f"{pair[0]} {pair[1]/total}")
                        left -= 1
        else:
            print(f"<unk> {final_prob['<unk>']}")
            print(f"\033[1mNext Predicted words are less than {k}.\033[0m")

    else:
        final_prob, lms_list, lambdas = load_model(f"./models/LM{lm_idx}_model.pkl")
        sorted_elements = {}

        for i in range(N):
            if i == N-1: 
                total_freq = lms_list[N-1-i]["TotalCnt"]
                for word, freq in lms_list[N-1-i].items():
                    if word == "TotalCnt" or word == "<unk>":
                        continue
                    if word in sorted_elements.keys():
                        sorted_elements[word] += (lambdas[N-1-i])*(freq/total_freq)
                    else:
                        sorted_elements[word] = (lambdas[N-1-i])*(freq/total_freq)
            elif tokens[i:] in lms_list[N-1-i].keys():
                total_freq = lms_list[N-1-i][tokens[i:]]["TotalCnt"]
                for word, freq in lms_list[N-1-i][tokens[i:]].items():
                    if word == "TotalCnt" :
                        continue
                    if word in sorted_elements.keys():
                        sorted_elements[word] += (lambdas[N-1-i])*(freq/total_freq)
                    else:
                        sorted_elements[word] = (lambdas[N-1-i])*(freq/total_freq)

        sorted_elements = sorted(sorted_elements.items(), key=lambda x:x[1], reverse=True)

        if len(sorted_elements) < k:
                for pair in sorted_elements:
                    print(f"{pair[0]} {pair[1]}")
                # print unkown probability 
                freq = lms_list[0]['<unk>']
                total_freq = lms_list[0]["TotalCnt"]
                print(f"<unk> {freq/total_freq}")
                print(f"\033[1mNext Predicted words are less than {k}.\033[0m")
        else:
            for pair in sorted_elements[:k]:
                print(f"{pair[0]} {pair[1]}")


if __name__ == "__main__":
    argc = len(sys.argv)

    if argc != 3 and argc != 4:
        print(f"There must be 3 or 4 arguments only.")
        sys.exit()

    txt = input("input sequence: ")
    print()

    if argc == 3:
        path = sys.argv[1]
        k = sys.argv[2]
        for i in range(10):
            print(f"\033[1mN={i+1}\033[0m")
            predict_word_before_smoothing(k, path, txt, i+1)
            print()
    else:
        smoothing_type = sys.argv[1]
        path = sys.argv[2]
        k = sys.argv[3]
        predict_word_after_smoothing(k, path, txt, smoothing_type)
