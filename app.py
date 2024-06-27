from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import joblib
import os
import time
import re
from nltk.util import clean_html
import pandas as pd
import numpy as np
#teste para render
import nltk
nltk.download('stopwords')
# metodo para substituir abreviacoes
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


app = Flask(__name__)
Bootstrap(app)

clf = joblib.load('model/svm.pkl') #load no modelo
count_vect = joblib.load('model/vect.pkl') # load no vetorizer (baw) 


@app.route('/')
def index():
    return render_template('index.html') #pagina html (front)


@app.route('/analyse', methods=['POST'])
def analyse():
    
	start = time.time()
	STOPWORDS = set(stopwords.words('english')) #load stopwords
	if request.method == 'POST':

		to_predict_list = request.form.to_dict()
		tweet = (to_predict_list['twitt'])

		new_tweet = tweet
		#new_tweet = pd.Series(tweet)

		new_tweet = re.sub(r'https?://\S+|www\.\S+', r'', new_tweet)  # remove URLS
		new_tweet = re.sub(r'<.*?>', r'', new_tweet)  # remove HTML
		new_tweet = BeautifulSoup(new_tweet, 'lxml').get_text()
		new_tweet = decontracted(new_tweet)
		new_tweet = re.sub(r'\d+', '', new_tweet).strip()  # remove number
		new_tweet = re.sub(r"[^\w\s\d]", "", new_tweet)  # remove pnctuations
		new_tweet = re.sub(r'@\w+', '', new_tweet)  # remove mentions
		new_tweet = re.sub(r'#\w+', '', new_tweet)  # remove hash
		new_tweet = re.sub(r"\s+", " ", new_tweet).strip()  # remove space
		new_tweet = re.sub("\S*\d\S*", "", new_tweet).strip()
		new_tweet = re.sub('[^A-Za-z]+', ' ', new_tweet) #lower case
		new_tweet = ' '.join([e.lower() for e in new_tweet.split() if e.lower() not in STOPWORDS]) #divide a string e procura stopwords e remove
		new_tweet = (new_tweet.strip())
		clean_tweet = (new_tweet) #variavel apenas para imprimir no console tweet formatado

		new_tweet = count_vect.transform([new_tweet])
		pred = clf.predict(new_tweet)
		prob = clf.predict_proba(new_tweet)

		       # DEF PREDICTION - vetoriza e classifica
		def prediction(sentence, vectorizer=count_vect, model=clf):
			x_input = vectorizer.transform(sentence).toarray()
			return model.predict_proba(x_input)

				
		saida = prediction([clean_tweet])
		indice_saida = str(np.argmax(saida))
		print("Saída da classe correta: ",indice_saida)

		sentiment = ''
		words = int (to_predict_list['range_words']) #pega valor do range slide da lista do post
		print("range de palavras: ",words)
		print("---"*20)
		print("A probabilidade para o twitter é de: ", prob)
		print("A análise de sentimento do twitter é para classe: ", pred)
		print("teste o que é prob[0][0]: ", prob[0][0])
		print("Tweet original: ", tweet,"\n")
		print("Tweet formatado: ",clean_tweet)
		print("---"*20)



		if indice_saida == "0":
			sentiment = "Negative sentiment"
		elif indice_saida == "1":
			sentiment = "Neutral sentiment"
		elif indice_saida == "2":			
			sentiment = "Positive sentiment"

 	    # LIME EXPLAIN
		class_names = ['NEGATIVE', 'NEUTRAL', 'POSITIVE'] #targets
		explainer = LimeTextExplainer(class_names=class_names)
		explanation = explainer.explain_instance(clean_tweet, prediction,  num_features=words, top_labels=1, labels=(0, 1, 2))
        #exp = explanation.save_to_file("templates/lime.html")
		resultado = explanation.as_html() # renderizado via javascript
		print("sentiment", sentiment)
		print("explainer LIME", explainer)

		end = time.time()
	final_time = end-start
	return render_template('index.html', prediction=sentiment, twitter=tweet, lime=resultado, final_time=final_time,range_value=words)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,debug=True)
    #desabilitar debug após encerrar cod. --,debug=True
