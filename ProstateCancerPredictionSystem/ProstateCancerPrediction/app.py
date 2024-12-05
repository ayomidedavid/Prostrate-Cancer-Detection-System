from flask import Flask, render_template, url_for, request
import pandas as pd, numpy as np
import pickle

# load the model from disk
filename = 'stacked_classifier_model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		# me = request.form['message']
		# message = [float(x) for x in me.split()]
		features = [
			float(request.form['radius']),
			float(request.form['texture']),
			float(request.form['perimeter']),
			float(request.form['area']),
			float(request.form['smoothness']),
			float(request.form['compactness']),
			float(request.form['symmetry']),
			float(request.form['fractal_dimension'])
		]
		vect = np.array(features).reshape(1, -1)
		my_prediction = clf.predict(vect)
	return render_template('home.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)

	
