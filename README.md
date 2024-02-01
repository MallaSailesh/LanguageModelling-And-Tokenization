# tokenizer.py

Tokenizer.py is a Python module that provides a customizable tokenizer class. The class allows you to tokenize sentences and words in a given text. Additionally, it includes functionality to identify and replace occurrences of Numbers, Mail IDs, Punctuation, URLs, Hashtags (#nlp), and Mentions (@sailesh).

## Features

- Tokenize sentences and words in a given text.
- Identify and replace the following occurrences:
  - Numbers replaced with `<NUM>`.
  - Mail IDs replaced with `<MAILID>`.
  - Punctuation identified.
  - URLs replaced with `<URL>`.
  - Hashtags (#nlp) replaced with `<HASHTAG>`.
  - Mentions (@sailesh) replaced with `<MENTION>`.

# n_gram.py

ngram.py is a Python module that provides a function which takes in the parameters - tokenized words of a text and n (in n-gram). 

- Calculate the count of occurrences of a word given a history.
- It also tracks the total number of occurrences of all word for that history in the "TotalCnt" i.e final_prob[history]["TotalCnt"] gives the total occurences of that history . This is used for calulating the probability.

# language_model.py

- Implemented Good turing smoothing. 
  - For training this on 1.txt -  *__python3 language_model.py g ./corpus/1.txt__*.  
  - It also prints the values of avg perplexity scores and perplexity scores in seperate folder called scores. 
  - The trained model is also stored in models folder . 
- Implemented Linear Interpolation
  - For training this on 1.txt -  *__python3 language_model.py i ./corpus/1.txt__*.
  - It also prints the values of avg perplexity scores and perplexity scores in seperate folder called scores. 
  - The trained model is also stored in models folder . 

# generator.py

Predict the next word for trained models of good turing , linear interpolation and without smoothing techniques. 

- Without smoothing
  - For running it *__python3 generator.py ./corpus/1.txt k__*.
- For good turing model
  - For running it *__python3 generator.py g ./corpus/1.txt k__*.
- For Linear Interpolation model
  - For running it *__python3 generator.py i ./corpus/1.txt k__*.

Note :- Here k can be any natural number . It gives the most probable k words. \
Note :- First train the models (run language_model.py) then generate(generator.py) because then onle models.py is full 
