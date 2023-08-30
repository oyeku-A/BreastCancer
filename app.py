import joblib
import numpy as np
from flask import Flask, render_template, request

model = joblib.load("models/model.pkl")

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home_page():
  form_data = {}
  if request.method == 'POST':
    clump_thickness = request.form['clump thickness']
    uniformity_of_cell_size = request.form['uniformity of cell size']
    uniformity_of_cell_shape = request.form['uniformity of cell shape']
    marginal_adhesion = request.form['marginal adhesion']
    single_epithelial_cell_size = request.form['single epithelial cell size']
    bare_nuclei = request.form['bare nuclei']
    bland_chromatin = request.form['bland chromatin']
    normal_nucleoli = request.form['normal nucleoli']
    mitoses = request.form['mitoses']
    form_data = request.form.to_dict()

    arr_inputs = np.array([[
      clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape,
      marginal_adhesion, single_epithelial_cell_size, bare_nuclei,
      bland_chromatin, normal_nucleoli, mitoses
    ]],
                          dtype='int64')

    prediction = model.predict(arr_inputs)
    return render_template('index.html', pred_=prediction, form_data=form_data)
  else:
    return render_template('index.html', form_data=form_data)


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)
