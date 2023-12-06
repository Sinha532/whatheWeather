from flask import Flask, render_template, request
import pickle
import numpy as np

model1=pickle.load(open('model1.sav',"rb"))
model2=pickle.load(open('model2.sav',"rb"))
model3=pickle.load(open('model3.sav',"rb"))
model4=pickle.load(open('model4.sav',"rb"))
model5=pickle.load(open('model5.sav',"rb"))

models={
    'Logistic Regression(0.83)':model1,
    'Decison Tree(0.72)':model2,
    'Random Forest(0.82)':model3,
    'Support Vector Machine(0.83)':model4,
    'Gradient Boosting(0.82)':model5
}

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html',model_options=models.keys())

@app.route('/predict',methods=['POST'])

def home():
    selected_model = request.form['selected_model']
    data1=float(request.form['a'])
    data2=float(request.form['b'])
    data3=float(request.form['c'])
    data4=float(request.form['d'])
    arr=np.array([[data1,data2,data3,data4]])

    model=models[selected_model]
    pred=model.predict(arr)

    return render_template("after.html",data=pred)

if __name__ =="__main__":
    app.run(debug=True)
    

