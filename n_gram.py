from tokenizer import MyTokenizer

# It accepts word tokenizer tokens and N as params
def generate_ngram_model(tokens, N):
    '''
    Gives the number of times a word comes, given history based on which 
    probability of its occurence can be calculated .
    '''
    ngram_model = {}
    for token in tokens:
        token = ["<start>"]*(N-1) + token + ["<end>"]

        ngrams = [tuple(token[i:i + N]) for i in range(len(token) - N + 1)]

        for gram in ngrams:
            prefix = tuple(gram[:-1])
            suffix = gram[-1]

            if prefix in ngram_model:
                if suffix in ngram_model[prefix]: 
                    ngram_model[prefix][suffix]+=1
                else:
                    ngram_model[prefix][suffix] = 1
                ngram_model[prefix]["TotalCnt"]+=1
            else:
                ngram_model[prefix] = {suffix:1}
                ngram_model[prefix]["TotalCnt"] = 1

    return ngram_model

if __name__ == "__main__":
    corpus_path = './corpus/sample.txt'
    N = 3

    with open(corpus_path, 'r', encoding='utf-8') as file:
        text = file.read()

    tokenizer = MyTokenizer()
    tokens = tokenizer.replace_tokens_with_placeholders(text)
    tokens =  tokenizer.tokenize(tokens)

    ngram_model = generate_ngram_model(tokens, N)
    print(ngram_model)
