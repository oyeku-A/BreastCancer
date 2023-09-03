import joblib
import numpy as np
from flask import Flask, render_template, request

model = joblib.load("models/model.pkl")

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home_page():
  form_data = {}
  if request.method == 'POST':
    form_data = request.form.to_dict()
    arr_inputs = np.array([[
      form_data['clump thickness'], form_data['uniformity of cell size'],
      form_data['uniformity of cell shape'], form_data['marginal adhesion'],
      form_data['single epithelial cell size'], form_data['bare nuclei'],
      form_data['bland chromatin'], form_data['normal nucleoli'],
      form_data['mitoses']
    ]],
                          dtype='int64')

    prediction = model.predict(arr_inputs)
    return render_template('index.html', pred_=prediction, form_data=form_data)
  else:
    return render_template('index.html', form_data=form_data)


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)
