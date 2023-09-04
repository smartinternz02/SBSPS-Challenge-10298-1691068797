# imports
from flask import Flask, render_template, request
import IBM_model as model

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("IBM_home.html")

@app.route("/layout")
def layout():
    return render_template("IBM_layout.html")

@app.route("/cover")
def cover():
    return render_template("cover.html")

@app.route("/result", methods = ['POST', 'GET'])
def result():
    #if request.method == 'GET':
     #   return "The URL /data is accessed directly. Try going to Home Page to submit form"
    if request.method == 'POST':
        form_data = request.form
        predicted_status = model.call_predict_placement(form_data)
        return render_template("result.html",predicted_status = predicted_status)

@app.route("/test")
def test():
    return render_template('IBM_test.html')

if __name__ == "__main__":
    app.run(debug=True)