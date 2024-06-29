from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)

with open('model.pkl','rb') as model_file,open('vectorizer.pkl','rb') as vectorizer_file:
    model=pickle.load(model_file)
    vectorizer=pickle.load(vectorizer_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    message=request.form["message"]
    data=[message]
    vect=vectorizer.transform(data).toarray()
    
    prediction=model.predict(vect)[0]
    
    probability=model.predict_proba(vect)[0][prediction]
    
    result={
        'prediction':'spam' if prediction==1 else 'ham',
        'probability':round(probability,3)*100
    }
    
    return jsonify(result)



if __name__=='__main__':
    app.run(debug=True)
    