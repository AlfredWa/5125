This is a Book recommand system using Flask-APP that use to connect with Dialogflow via ngrok

Before you can run this application, you'll need the following installed:
- Python 3.x
- Flask
- Ngrok
- VScode

python libraries:
pip install flask, google.cloud, re, pandas, scikit-learn, nltk, joblib, scipy, pickle-mixin, matplotlib, seaborn

Try the chatbot on Diagflow
https://dialogflow.cloud.google.com/#/agent/rock-sorter-427520-c5/intents

To run the backend application, run the 5125project.py with vscode or use command "python3 5125project.py"

Use ngrok to expose the local server, download ngrok then open ngrok and run command "ngrok http 5125", copy the HTTPS URL from ngrok the paste the url and add "/webhook" after the url to the Webhook fulfillment of Dialogflow.