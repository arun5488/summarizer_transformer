from src.summarizer import logger
from dotenv import load_dotenv
from src.summarizer import constants as const
from src.summarizer.pipeline.text_summarization_piepline import TextSummarizationPipeline
from flask import Flask, render_template, request
import os

load_dotenv(".flaskenv")
app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def index():
    try:
        logger.info("inside index method")
        if request.method == 'POST':
            logger.info("Inside POST method")
            article = request.form.get('article')
            if len(article)>0:
                logger.info("article received successfully")
                summary = TextSummarizationPipeline().initiate_text_summarization_pipeline(article)
                return render_template('index.html', article = article, summary = summary)
            else:
                logger.info("aritcle not received")   
                return render_template('index.html')             
        else:
            logger.info("inside GET method")
            return render_template('index.html')
    except Exception as e:
        logger.error(f"Error occured inside index method:{e}")
        raise e


if __name__ == "__main__":
    try:
        logger.info("launching textsummarizer app")
        flask_env = os.getenv('FLASK_ENV')
        logger.info(f"Flask env type:{flask_env}")
        flask_host = os.getenv('FLASK_RUN_HOST')
        logger.info(f"Flask host:{flask_env}")
        flask_port = os.getenv('FLASK_RUN_PORT')
        app.run(host=flask_host, port=flask_port, debug= True)
    except Exception as e:
        logger.error(f"Error occured inside main in app.py:{e}")
        raise e