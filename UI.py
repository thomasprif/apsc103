from flask import Flask, render_template, request 
import os

app = Flask(__name__)

picFolder = os.path.join('static', 'pics')

app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/color', methods=['POST'])
def color():
    color = request.form['color']
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'hyperlooptunnel.jpg')
    return render_template('base.html', color=color, user_image=pic1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sensitivity = request.form['sensitivity']
        return 'Sensitivity updated!'
    else:
        current_sensitivity = 50
        return render_template('base.html', current_sensitivity=current_sensitivity)


if __name__ == '__main__':
    app.run(debug=True)

#END OF PYTHON CODE

