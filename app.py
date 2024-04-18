from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics
app = Flask(__name__, static_url_path='/static', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    #bp = int(request.form['bp'])
    bp_limit=int(request.form['bp_limit'])
    sg=int(request.form['sg'])
    al=int(request.form['al'])
    rbc=int(request.form['rbc'])
    su=int(request.form['su'])
    pc=int(request.form['pc'])
    #pcc=int(request.form['pcc'])
    #ba=int(request.form['ba'])
    bgr=int(request.form['bgr'])
    #bu=int(request.form['bu'])
    sod=int(request.form['sod'])
    sc=int(request.form['sc'])
    #pot=int(request.form['pot'])
    hemo=int(request.form['hemo'])
    pcv=int(request.form['pcv'])
    rbcc=int(request.form['rbcc'])
    wbcc=int(request.form['wbcc'])
    htn=int(request.form['htn'])
    dm=int(request.form['dm'])
    cad=int(request.form['cad'])
    appet=int(request.form['appet'])
    #pe=int(request.form['pe'])
    ane=int(request.form['ane'])
    gfr=int(request.form['gfr'])
    age=int(request.form['age'])

    df=pd.read_csv('preprocessed.csv')
    X=df.drop(columns=["stage"])
    y=df["stage"]
    classifier=XGBClassifier()
    rfe=RFE(estimator=classifier,n_features_to_select=20,step=1)
    rfe.fit(X,y)
    selected_features=X.columns[rfe.support_]
    X_selected = X[selected_features]

    print(X_selected)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=30)
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X_train, y_train)
    data = [[bp_limit,sg,al,rbc,su,pc,bgr,sod,sc,hemo,pcv,rbcc,wbcc,htn,dm,cad,appet,ane,gfr,age]]
    new_df=pd.DataFrame(data, columns=['bp limit','sg','al','rbc','su','pc','bgr','sod','sc','hemo','pcv','rbcc','wbcc','htn','dm','cad','appet','ane','grf','age'])
    result = xgb_classifier.predict(new_df)
    #predicted_class = result.argmax()
    
    if result==0:
        output = "The patient does not have CKD."
        
    elif result==1:
        output = "The patient is diagnosed with CKD.\nThe disease is in the 1st stage."
    
    elif result==2:
        output = "The patient is diagnosed with CKD.\nThe disease is in the 2nd stage."
        
    elif result==3:
        output = "The patient is diagnosed with CKD.\nThe disease is in the 3rd stage."
    
    elif result==4:
        output = "The patient is diagnosed with CKD.\nThe disease is in the 4th stage."
        
    elif result==5:
        output = "The patient is diagnosed with CKD.\nThe disease is in the 5th stage."
    return jsonify({'result': output})

if __name__ == '__main__':
    app.run(debug=True)
