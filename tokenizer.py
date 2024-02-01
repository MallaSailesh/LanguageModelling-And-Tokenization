import re

class MyTokenizer:
    def __init__(self):
        # self.sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=[\.|\?|\!])\s+|\n+'
        self.sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=[\.|\?|\!])\s+' 
        self.word_pattern =   r'\b\w+\b|[!-/:-@[-`{-~\n]'
        # self.word_pattern = r'\b\w+\b'
        self.number_pattern = r'\b\d+\b'  
        self.mail_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.punctuation_pattern = r'[!-/:-@[-`{-~]'
        self.url_pattern = r'(http://|ftp://|https://)([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'
        self.hashtag_pattern = r'\#\w+'
        self.mention_pattern = r'\@\w+'
        
    def tokenize_sentences(self, text):
        return re.split(self.sentence_pattern, text)

    def tokenize_words(self, sentence):
        # sentence = sentence.lower()
        words = re.findall(self.word_pattern, sentence)
        length = len(words)
        cnt = 0
        modified_words = []
        for i in range(length):
            word = words[i]
            if word == ">" and i-1 >= 0 and (words[i-1] == "HASHTAG" or words[i-1] == "MENTION" or words[i-1] == "URL" or words[i-1] == "MAILID" or words[i-1] == "NUM"):
                if i-2 >= 0 and words[i-2] == "<" :
                    word = ''.join(words[i-2:i+1])
                    modified_words.pop()
                    modified_words.pop()
            modified_words.append(word)
        i = 0
        length = len(modified_words)
        updated_words = []
        while i < length:
            word =  modified_words[i]
            if word == "'" and i+1 < length and modified_words[i+1] == "s":
                word = ''.join(modified_words[i:i+2])
                i+=1
            elif word == "'" and cnt%2 == 0:
                cnt+=1
                word = ''.join(modified_words[i:i+2])
                i+=1
            elif word == "'" and cnt%2:
                cnt+=1
                word = ''.join(modified_words[i-1:i+1])
                updated_words.pop()
            updated_words.append(word)
            i+=1
        return updated_words
    
    def tokenize_numbers(self, text):
        return re.findall(self.number_pattern, text)
    
    def tokenize_mail_ids(self, text):
        return re.findall(self.mail_pattern, text)
    
    def tokenize_urls(self, text):
        urls = re.findall(self.url_pattern, text)
        modified_urls = []
        for url in urls:
            url = ''.join(url)
            modified_urls.append(url)
        return modified_urls
    
    def tokenize_hashtags(self, text):
        return re.findall(self.hashtag_pattern, text)
    
    def tokenize_mentions(self, text):
        return re.findall(self.mention_pattern, text)

    def tokenize_punctuation(self, text):
        return re.findall(self.punctuation_pattern, text)
    
    def tokenize(self, text):
        return [self.tokenize_words(sentence) for sentence in re.split(self.sentence_pattern, text)]
    
    def replace_tokens_with_placeholders(self, text):
        # Replace tokens with placeholders
        text = re.sub(self.url_pattern, '<URL>', text)
        text = re.sub(self.hashtag_pattern, '<HASHTAG>', text)
        text = re.sub(self.mention_pattern, '<MENTION>', text)
        text = re.sub(self.number_pattern, '<NUM>', text)
        text = re.sub(self.mail_pattern, '<MAILID>', text)
        return text
    
if __name__ == "__main__":

    text = input("your text: ")
    # with open('./corpus/sample.txt', 'r', encoding='utf-8') as f:
    #     text  = f.read()
 
    tokenizer = MyTokenizer()
    tokenized_text = tokenizer.replace_tokens_with_placeholders(text)
    tokenized_text = tokenizer.tokenize(tokenized_text)

    print(f"tokenized text: {tokenized_text}")

