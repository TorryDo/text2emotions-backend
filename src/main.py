import time

from fastapi import FastAPI

from src.my_model import MyModel

app = FastAPI()


@app.get("/")
def read_root():
    return {
        "author": "Tri Do Nguyen",
        "github": "https://github.com/torrydo",
        "github_repository": "https://github.com/torrydo/text2emotions_fastapi",
        "model": "rnn",
        "api": "/docs"
    }


my_model = MyModel()


@app.get("/v1/get_emotions/")
def text_to_emotions(sentence: str, max_label: int = 2):

    time_start = time.time()
    predictions = my_model.predicts(sentence)
    time_consumed = time.time() - time_start

    predictions = sorted(predictions, key=lambda tp: tp[1], reverse=True)[:max_label]

    result = []
    for label, probability in predictions:
        label_prob = {
            'label': label,
            'probability': probability
        }
        result.append(label_prob)
    return {
        "time": time_consumed,
        "result": result
    }
