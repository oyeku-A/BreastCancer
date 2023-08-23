from flask import Flask

app = Flask(__name__)

@app.route("/")
def home_page():
  return "Work in progress"

if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)
