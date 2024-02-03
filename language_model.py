import sys, random, math
import pickle, os
import numpy as np
from sklearn.linear_model import LinearRegression
from tokenizer import MyTokenizer
from n_gram import generate_ngram_model

def save_model(model, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(model ,f) 

# N > 1
def LM_GoodTuring(train_data, test_data, lm_idx, N=3):

    tokenizer = MyTokenizer()
    train_tokens = [tokenizer.tokenize_words(sentence) for sentence in train_data]
    test_tokens = [tokenizer.tokenize_words(sentence) for sentence in test_data]
    
    lm = generate_ngram_model(train_tokens, N)
    num_words = 0
    Trigram_Cnt = {}
    for key, value in lm.items():
        for keySub, valueSub in value.items():
            if keySub == "TotalCnt":
                continue
            trigram = []
            for k in key : 
                trigram.append(k)
            trigram.append(keySub)
            trigram = tuple(trigram) 
            if trigram in Trigram_Cnt.keys():
                Trigram_Cnt[trigram]+=valueSub
            else:
                Trigram_Cnt[trigram]=valueSub
            num_words += valueSub


    # Nr caclualtion 
    Nr = {} ; Zr = {} ; R = []
    for trigram, r in Trigram_Cnt.items():
        if r in Nr.keys():
            Nr[r] += 1
            Zr[r] += 1
        else:
            Nr[r] = 1
            Zr[r] = 1
            R.append(r)

    R = sorted(R)

    # Smoothing Nr - q r t in sequence 
    q = [0] ; t = []

    for i in range(len(R)-1):
        q.append(R[i])
        t.append(R[i+1])

    # Assuming there is some text provided 
    t.append(2*R[len(R)-1])
    q[len(q)-1] *= 2 # for handling the special case 

    # Calculation of Zr
    for i in range(len(R)):
        Zr[R[i]] *= 2
        Zr[R[i]] /= (t[i]-q[i])

    log_r = np.log(np.array(list(Zr.keys())).reshape(-1, 1))
    log_Zr = np.log(np.array(list(Zr.values())))

    model = LinearRegression()
    model.fit(log_r, log_Zr)

    a = model.intercept_
    a = math.exp(a)
    b = model.coef_[0]

    final_prob = {}

    for trigram, r in Trigram_Cnt.items():
        pr = ((r+1)**(b+1))/(num_words*(r**b))
        if trigram[:-1] not in final_prob.keys():
            final_prob[trigram[:-1]] = {trigram[N-1]: pr}
            final_prob[trigram[:-1]]["TotalProb"] = pr
        else:
            final_prob[trigram[:-1]][trigram[N-1]] = pr
            final_prob[trigram[:-1]]["TotalProb"] += pr

    final_prob['<unk>'] = (Nr[1]**(1/N))/(num_words)   
    # final_prob['<unk>'] = a/(num_words)

    train_file_path = "./scores/2021101106_LM"+str(lm_idx)+"_train-perplexity.txt"
    test_file_path = "./scores/2021101106_LM"+str(lm_idx)+"_test-perplexity.txt"

    with open(train_file_path, 'w', encoding='utf-8') as f:
        perplexity_scores_train = []
    
        for token in train_tokens:
            token = ["<start>"]*(N-1) + token + ["<end>"]
            ngrams = [tuple(token[i:i + N]) for i in range(len(token) - N + 1)]
            score = 1
            for gram in ngrams:
                val = (final_prob[gram[:-1]]["TotalProb"]/final_prob[gram[:-1]][gram[-1]])
                val **= (1/(len(ngrams)))
                score *= val
            f.write(' '.join(token)+f"   {score}\n")
            perplexity_scores_train.append(score)
    with open(train_file_path, 'r+', encoding='utf-8') as f:
        prev_content = f.read()
        f.seek(0, 0)
        f.write(f"Avg Perplexity Score on train corpus : {sum(perplexity_scores_train)/len(perplexity_scores_train)}\n"+prev_content)
    # print(sum(perplexity_scores_train)/len(perplexity_scores_train))
    
    with open(test_file_path, 'w', encoding='utf-8') as f:
        perplexity_scores_test = []

        for token in test_tokens:
            token = ["<start>"]*(N-1) + token + ["<end>"]
            ngrams = [tuple(token[i:i + N]) for i in range(len(token) - N + 1)]
            score = 1
            for gram in ngrams:
                if gram[:-1] not in final_prob.keys():
                    val = (1/final_prob['<unk>'])
                elif gram[-1]  not in final_prob[gram[:-1]].keys():
                    val = (1/final_prob['<unk>'])
                else:
                    val = (final_prob[gram[:-1]]["TotalProb"]/final_prob[gram[:-1]][gram[-1]])
                val **= (1/len(ngrams))
                score *= val
            f.write(' '.join(token)+f"   {score}\n")
            perplexity_scores_test.append(score)

    with open(test_file_path, 'r+', encoding='utf-8') as f:
        prev_content = f.read()
        f.seek(0, 0)
        f.write(f"Avg Perplexity Score on test corpus : {sum(perplexity_scores_test)/len(perplexity_scores_test)}\n"+prev_content)
    # print(sum(perplexity_scores_test)/len(perplexity_scores_test))

    return final_prob

def LM_Interpolation(train_data, test_data, lm_idx, N=3):

    tokenizer = MyTokenizer()
    train_tokens = [tokenizer.tokenize_words(sentence) for sentence in train_data]
    test_tokens = [tokenizer.tokenize_words(sentence) for sentence in test_data]

    lms_list = []; lambdas = [0]*N
    single_words = 0

    for i in range(N):
        if i == 0:
            lm = {}
            for token in train_tokens: 
                token = token + ["<end>"]
                for word in token:
                    single_words+=1
                    if word not in lm.keys():
                        lm[word] = 1
                    else:
                        lm[word]+=1
            lm["TotalCnt"] = single_words
            # Convert the low frequency words into unknown words 
            unk_words = []
            for word in lm.keys():
                if lm[word] < 2:
                    unk_words.append(word)
            lm["<unk>"] = 0
            for word in unk_words:
                lm["<unk>"] += lm[word]
                del lm[word]
        else:
            lm = generate_ngram_model(train_tokens, i+1)
        lms_list.append(lm)
    
    # Calculate the lambda values 
    for history, words in lms_list[N-1].items():
        for word in words:
            if word == "TotalCnt":
                continue
            trigram = list(history) ; trigram.append(word) ;    trigram  = tuple(trigram)
            maxv = 0 ; ind = N-1
            for i in range(N-2):
                num = lms_list[N-1-i][trigram[:-1]][trigram[-1]]-1
                if trigram[:-2] in lms_list[N-2-i].keys() and trigram[-2] in lms_list[N-2-i][trigram[:-2]].keys():
                    den = lms_list[N-2-i][trigram[:-2]][trigram[-2]]-1
                else:
                    den = -1
                if den == 0:
                    val = 0
                else:
                    val  = num/den
                if maxv < val:
                    ind = N-1-i
                    maxv = val
                trigram = trigram[1:]
            # Edge Case 1
            num = lms_list[1][trigram[:-1]][trigram[-1]]-1
            if trigram[-2] not in lms_list[0].keys():
                den = lms_list[0]['<unk>']-1
            else:
                den = lms_list[0][trigram[-2]]-1
            if den == 0:
                val = 0 
            else:
                val = num/den
            if maxv < val:
                ind = 1
                maxv = val
            # Edge Case 2
            if word not in lms_list[0].keys():
                num = lms_list[0]['<unk>']-1
            else:
                num = lms_list[0][word]-1
            den = single_words-1
            if den == 0:
                val = 0 
            else:
                val = num/den
            if maxv < val:
                ind = 0
                maxv = val
            
            # Finally update the highest corresponding lambda value
            lambdas[ind] += lms_list[N-1][history][word]
    
    lambdas = [x/sum(lambdas) for x in lambdas]

    final_prob = {}

    for history, words in lms_list[N-1].items():
        final_prob[history] = {}
        for word in words:
            if word != "TotalCnt":
                final_prob[history][word] = 0
                for i in range(N):
                    if i == 0 and word in lms_list[0].keys():
                        final_prob[history][word] += ((lambdas[i]*lms_list[i][word])/(single_words))
                    elif i==0:
                        final_prob[history][word] += ((lambdas[i]*lms_list[i]['<unk>'])/(single_words))
                    else:
                        final_prob[history][word] += ((lambdas[i]*lms_list[i][history[N-i-1:]][word])/(lms_list[i][history[N-i-1:]]["TotalCnt"]))

    train_file_path = "./scores/2021101106_LM"+str(lm_idx)+"_train-perplexity.txt"
    test_file_path = "./scores/2021101106_LM"+str(lm_idx)+"_test-perplexity.txt"

    with open(train_file_path, 'w', encoding='utf-8') as f:
        perplexity_scores_train = []

        for token in train_tokens:
            token = ["<start>"]*(N-1) + token + ["<end>"]
            ngrams = [tuple(token[i:i + N]) for i in range(len(token) - N + 1)]
            score = 1
            for gram in ngrams:
                val = (1/final_prob[gram[:-1]][gram[-1]])
                val **= (1/len(ngrams))
                score *= val
            f.write(' '.join(token)+f"   {score}\n")
            perplexity_scores_train.append(score)

    with open(train_file_path, 'r+', encoding='utf-8') as f:
        prev_content = f.read()
        f.seek(0, 0)
        f.write(f"Avg Perplexity Score on train corpus : {sum(perplexity_scores_train)/len(perplexity_scores_train)}\n"+prev_content)

    # print(sum(perplexity_scores_train)/len(perplexity_scores_train))
    
    with open(test_file_path, 'w', encoding='utf-8') as f:
        perplexity_scores_test = [] 

        for token in test_tokens:
            token = ["<start>"]*(N-1) + token + ["<end>"]
            ngrams = [tuple(token[i:i + N]) for i in range(len(token) - N + 1)]
            score = 1
            for gram in ngrams:
                if gram in final_prob.keys():
                    val = (1/final_prob[gram[:-1]][gram[-1]])
                else:
                    n = N-2
                    val = 0
                    while n>1:
                        if gram[:n-N] in lms_list[n].keys():
                            if gram[:-1] in lms_list[n][gram[:n-N]].keys():
                                val += (((lambdas[n])*lms_list[n][gram[:n-N]][gram[-1]])/(lms_list[n][gram[:n-N]]["TotalCnt"]))
                        n-=1
                    if gram[-1] in lms_list[0].keys():
                        val += lambdas[0]*(lms_list[0][gram[-1]]/lms_list[0]["TotalCnt"])
                    else:
                        val += lambdas[0]*(lms_list[0]["<unk>"]/lms_list[0]["TotalCnt"])
                    val = 1/val
                val **= (1/len(ngrams))
                score *= val
            f.write(" ".join(token)+f"   {score}\n")
            perplexity_scores_test.append(score)
    
    with open(test_file_path, 'r+', encoding='utf-8') as f:
        prev_content = f.read()
        f.seek(0, 0)
        f.write(f"Avg Perplexity Score on test corpus : {sum(perplexity_scores_test)/len(perplexity_scores_test)}\n"+prev_content)

    # print(sum(perplexity_scores_test)/len(perplexity_scores_test))

    return final_prob, lms_list, lambdas

def calc_score(final_prob, txt, lms_list=None, lambdas=None, N=3):

    tokenizer = MyTokenizer()
    txt = tokenizer.replace_tokens_with_placeholders(txt)
    txt_tokens = tokenizer.tokenize(txt)

    scores = []
    if lms_list == None:
        for token in txt_tokens:
            token = ["<start>"]*(N-1) + token + ["<end>"]
            ngrams = [tuple(token[i:i + N]) for i in range(len(token) - N + 1)]
            score = 1
            for gram in ngrams:
                if gram[:-1] not in final_prob.keys():
                    score *= (final_prob['<unk>'])
                elif gram[-1]  not in final_prob[gram[:-1]].keys():
                    score *= (final_prob['<unk>'])
                else:
                    prob = final_prob[gram[:-1]][gram[-1]]
                    total_prob = final_prob[gram[:-1]]["TotalProb"]
                    score *= (prob/total_prob)
    else:
        for token in txt_tokens:
            token = ["<start>"]*(N-1) + token + ["<end>"]
            ngrams = [tuple(token[i:i + N]) for i in range(len(token) - N + 1)]
            score = 1
            for gram in ngrams:
                if gram in final_prob.keys():
                    val = (final_prob[gram[:-1]][gram[-1]])
                else:
                    n = N-2
                    val = 0
                    while n>1:
                        if gram[:n-N] in lms_list[n].keys():
                            if gram[:-1] in lms_list[n][gram[:n-N]].keys():
                                val += (((lambdas[n])*lms_list[n][gram[:n-N]][gram[-1]])/(lms_list[n][gram[:n-N]]["TotalCnt"]))
                        n-=1
                    if gram[-1] in lms_list[0].keys():
                        val += lambdas[0]*(lms_list[0][gram[-1]]/lms_list[0]["TotalCnt"])
                    else:
                        val += lambdas[0]*(lms_list[0]["<unk>"]/lms_list[0]["TotalCnt"])
                score *= val

    print(f"score: {score}")

def split_train_and_test(path):
    tokenizer = MyTokenizer()
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = tokenizer.replace_tokens_with_placeholders(text)
    sentences = tokenizer.tokenize_sentences(text)
    number_of_test_indices = 1000
    test_indices = random.sample(range(len(sentences)), number_of_test_indices)
    train_sent = [sent for idx, sent in enumerate(sentences) if idx not in test_indices]
    test_sent = [sentences[idx] for idx in test_indices]
    return train_sent, test_sent

if __name__ == "__main__":
    argc = len(sys.argv)

    if(argc != 3) :
        print(f"There must be 3 arguments only.")
        sys.exit()

    smoothing_type = sys.argv[1]
    path = sys.argv[2]

    train_sent, test_sent = split_train_and_test(path)

    lm_idx = 1
    if smoothing_type == 'g' and path == './corpus/2.txt':
        lm_idx = 3
    if smoothing_type == 'i' and path == './corpus/1.txt':
        lm_idx = 2
    if smoothing_type == 'i' and path == './corpus/2.txt':
        lm_idx = 4

    txt = input("input sequence: ")
    
    if smoothing_type == 'g':
        final_prob = LM_GoodTuring(train_sent, test_sent, lm_idx)
        save_model(final_prob, f'./models/LM{lm_idx}_model.pkl')
        calc_score(final_prob, txt)
    elif smoothing_type == 'i':
        final_prob, lms_list, lambdas = LM_Interpolation(train_sent, test_sent, lm_idx)
        save_model([final_prob, lms_list, lambdas], f'./models/LM{lm_idx}_model.pkl')
        calc_score(final_prob, txt, lms_list, lambdas)
    else:
        print("Argument 2 must be 'g' or 'i'.")
        sys.exit()


