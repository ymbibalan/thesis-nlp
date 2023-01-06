import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers.utils import logging
logging.set_verbosity_error()
pd.set_option('display.max_colwidth', None)

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


FILE_PATH = '/Users/yousef/code/thesis-nlp/dataset/incident-ohlc-v06-scalar.csv'
COLUMNS = ['w_open', 'w_high', 'w_low', 'w_close', 'new', 'started',
           'done', 'dayofweek', 'dayofmonth', 'dayofyear', 'target', 'y_ystrdy']

# N_ROWS = 100
# df_ = pd.read_csv(FILE_PATH,parse_dates=['date'],index_col=['date'], nrows=N_ROWS)
df_ = pd.read_csv(FILE_PATH, parse_dates=['date'], index_col=['date'])
df_ = (df_.sort_index())[COLUMNS]
df_ = df_.iloc[1:-1]  # Remove NaN

max_number_in_data = int(df_.describe().transpose()["max"].max())
print(f'Max number in the data is {max_number_in_data}')
# display(df_.head(2))

df = pd.DataFrame()
df['label'] = df_['target']
df['text'] = ''
for col in COLUMNS[:-2]:
    df_[col] = df_[col].apply(lambda x: f'{x:.0f}')
    df['text'] = df['text'] + f' {col} ' + df_[col].astype(str)

# display(df.head(2))


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
for x in COLUMNS:
    tokenizer.add_tokens(x)

for x in range(max_number_in_data):
    tokenizer.add_tokens(str(x))


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from datasets import Dataset


df_lb = pd.DataFrame()
df_lb['label'] = ''
df_lb['text'] = ''

LOOKBACK_WINDOW = 5
for i in range(LOOKBACK_WINDOW, len(df)):
    text = ' '.join(df['text'][i - LOOKBACK_WINDOW:i].to_list())
    label = df['label'][i]
    df_lb.loc[len(df_lb.index)] = [label, text]

datasets = []
N_SPLIT = 10
tscv = TimeSeriesSplit(n_splits=N_SPLIT)
for train_index, test_index in tscv.split(df_lb):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    df_tr = df_lb.iloc[train_index]
    df_ts = df_lb.iloc[test_index]
    tr_ds = Dataset.from_pandas(df_tr, split="train", preserve_index=False)
    ts_ds = Dataset.from_pandas(df_ts, split="test", preserve_index=False)
    datasets.append({'train': tr_ds, 'test': ts_ds})

print(f'length of datasets is {len(datasets)}')
print({datasets[0]['train']})
print({datasets[0]['test']})


from datasets import load_metric
from sklearn.metrics import mean_squared_error
import evaluate


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # rmse = mean_squared_error(labels, predictions, squared=False)
    rmse = mean_squared_error(labels, predictions, squared=True)
    return {"rmse": rmse}


def compute_metrics_mape(eval_pred):
    predictions, labels = eval_pred
    mape = mape_metric.compute(predictions=predictions, references=labels)
    return {"mape": mape}

    # num_labels =1 means regression
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1)
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(output_dir="test_trainer",
                                  logging_strategy="epoch",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  num_train_epochs=3,
                                  save_total_limit=2,
                                  save_strategy='no',
                                  load_best_model_at_end=False,
                                  report_to="none",
                                  optim="adamw_torch",
                                  # optim="adagrad"


                                  )


train_output = []
test_output = []

transformers.utils.logging.set_verbosity_error()

for idx, ds in enumerate(datasets):
    tokenized_train_ds = ds['train'].map(tokenize_function, batched=True)
    tokenized_test_ds = ds['test'].map(tokenize_function, batched=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_test_ds,
        compute_metrics=compute_metrics
    )
    print(f'\n<= Trainng {idx} of {len(datasets)} .... =>')
    trainer.train()
    train_output.append(trainer.evaluate(tokenized_train_ds))
    test_output.append(trainer.evaluate())
print('\n<= Train is done =>')

print(f'key\t\t train.metric \t test.metrics')

for key in list(predictions_test.metrics.keys())[:2]:
    print(
        f'{key} \t\t{predictions_train.metrics[key]:0.3f} \t\t{predictions_test.metrics[key]:0.3f}')

    # save the model/tokenizer

model.save_pretrained("model")
tokenizer.save_pretrained("tokenizer")
