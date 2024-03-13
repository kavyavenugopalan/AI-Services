# Import required libraries

from flask import Flask, request, render_template, session, Response
import subprocess
import json

# Dictionary to store uploaded file names
uploadedFiles = {}  

# Flask application
app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('web.html')

@app.route('/upload', methods=['POST'])
def upload_ref_file():
    if 'pdfFile' not in request.files :
        return 'No file part'

    pdfFile = request.files['pdfFile']


    if pdfFile.filename == '' : 
      return render_template('web.html')

    # Retrieve file names
    pdfFilename = pdfFile.filename
    
    
    
  

    # Store file names in the dictionary
    uploadedFiles['pdfFilename'] = pdfFilename
    pdfFile.save(pdfFilename)
    # run embedding.py
    #subprocess.run(["python", "embedding.py", pdfFilename], check=True,capture_output=True, text=True)
    try:
        subprocess.run(["python", "embedding.py", pdfFilename], check=True,capture_output=True, text=True)
        return Response(status=204)  
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error output: {e.output}")
      
    


@app.route('/uploadquest', methods=['POST'])
def upload_quest_file():
    if 'questionSetFile' not in request.files:
        return 'No file part'

    
    questionSetFile = request.files['questionSetFile']
   

    if questionSetFile.filename == '':
        return render_template('web.html')

    # Retrieve file names
   
    questionSetFilename = questionSetFile.filename
    uploadedFiles['questionSetFilename'] = questionSetFilename

    # You can save the files to a directory or process them as needed
    # For example, saving the files:
   
    questionSetFile.save(questionSetFilename)
    

  
    # run generator.py
    subprocess.run(["python", "generator.py", questionSetFilename], check=True, text=True)
    return Response(status=204)  



@app.route('/evaluate', methods=['POST'])
def evaluate_files():

    if 'answerSheetFile' not in request.files :
        return 'No file part'

    answerSheetFile = request.files['answerSheetFile']
    uploadedFiles['answerSheetFile'] = answerSheetFile

    if answerSheetFile.filename == '' : 
        return render_template('web.html')

    # Retrieve file names
    answerSheetFilename = answerSheetFile.filename
    questionSetFilename = uploadedFiles['questionSetFilename']
    answerSheetFile.save(answerSheetFilename)
   
    # Run similarity.py
    try:
        subprocess.run(["python", "similarity.py", answerSheetFilename,questionSetFilename], check=True,capture_output=True, text=True)
        resultData = json.load(open('result.json'))
        return render_template('result.html', result=resultData)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error output: {e.output}")
    


if __name__ == '__main__':
    app.run(debug=True)
  
