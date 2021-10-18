from flask import Flask, render_template, request, redirect
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["POST","GET"])
def index():
    if request.method == "POST":
        LR = joblib.load("LR1.pkl")
        if request.form["yoe"] == "":
            return render_template("index.html", result="MAA CHUDA")
        y_pred = LR.predict(np.array(int(request.form['yoe'])).reshape(-1, 1))
        y_pred = round(y_pred[0], 2)
        return render_template("index.html", result=y_pred)
    else: 
        return render_template("index.html", result=0)

if __name__ == "__main__":
    app.run(debug=True)
