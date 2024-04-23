import math as math
from flask import Flask, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/users', methods=['GET'])
def get_data():
    data =   [
            {
                'name': 'John Doe',
                'email': 'johndoe@example.com',
            }
        ];
    return jsonify(data)
  
@app.route('/api/v1/team', methods=['GET'])
def get_data_team():
    team_data =   [
            {
                'name': 'M Nouman Ejaz',
                'email': 'noumanejaz92@gmail.com',
                'description': 'Expert in the Web technologies and Machine learning!',
            },      
            {
                'name': 'Shahid Chaudhary',
                'email': 'shahidchaudhary0729@gmail.com',
                'description': 'Expert in the Web technologies!',
            },
            {
                'name': 'Salah Ud Din',
                'email': 'sallujutt33@gmail.com',
                'description': 'Expert in the Data Science!',
            },
            {
                'name': 'Khaula Sohail',
                'email': 'Khaulasohail313@gmail.com',
                'description': 'Expert in the Mobile development and Data Science!',
            },
             
        ];
    return jsonify(team_data)
  

@app.route('/api/v1/ping', methods=['GET'])
def pingServer():    
    return jsonify({'code': '200', 'message': 'Successfully connected with flask!'})
   

if __name__ == "__main__":
    app.run(debug=True)