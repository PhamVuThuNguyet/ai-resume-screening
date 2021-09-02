import gensim
import glob
import pandas as pd
from Preprocessing import Preprocessing
from PDF2Txt import convert_data_to_pdf, ConvertPDFtoText
from typing import List
import os
from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances
import streamlit as st
import logging
import psutil
import shutil
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tcn import TCN
import pickle
import tensorflow as tf


def cal_cosine_score(model, sentences1: List[str], sentences2: List[str],
                     batch_size: int = 8, show_progress_bar: bool = False):
    """
    :param show_progress_bar:
    :param batch_size:
    :param model:
    :param sentences1: List with the first sentence in a pair
    :param sentences2: List with the second sentence in a pair
    """
    embeddings1 = model.encode(sentences1, batch_size=batch_size,
                               show_progress_bar=show_progress_bar, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, batch_size=batch_size,
                               show_progress_bar=show_progress_bar, convert_to_numpy=True)

    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    return cosine_scores.tolist()


def preprocessing_text(uploaded_file_path, pdf_path, txt_path, preprocessed_path):
    print(uploaded_file_path)
    convert_data_to_pdf.convert_to_pdf(data_path=uploaded_file_path, pdf_output_dir=pdf_path)
    pdfs_path_list = ConvertPDFtoText.get_all_pdf_paths(pdf_path)
    os.makedirs(txt_path, exist_ok=True)
    ConvertPDFtoText.convert_pdf_to_txt(pdfs_path_list, txt_path)
    txt_path_list = Preprocessing.get_all_txt_paths(txt_path)
    os.makedirs(preprocessed_path, exist_ok=True)
    Preprocessing.preprocessing(txt_path_list, preprocessed_path)


def print_memory_usage():
    logging.info(f"RAM memory % used: {psutil.virtual_memory()[2]}")


def save_uploaded_file(uploaded_file, api=False):
    os.makedirs("./data/uploaded_file", exist_ok=True)
    with open(os.path.join("./data/uploaded_file", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    if not api:
        return st.success("Saved File:{} to ./data/uploaded_file".format(uploaded_file.name))


def save_uploaded_file_fast_api(uploaded_file):
    os.makedirs("./data/uploaded_file", exist_ok=True)
    with open(os.path.join("./data/uploaded_file", uploaded_file.filename), "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)


def get_tcn_model(model_dir, tokenizer_dir):
    with open(tokenizer_dir, 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = keras.models.load_model(model_dir, custom_objects={'TCN': TCN})

    return model, tokenizer


def get_txt_list(text_path):
    txt_text, txt_name = [], []
    for filename in glob.glob(os.path.join(text_path, '*.txt')):
        txt_name.append(filename.split("\\")[-1].split(".")[0])
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            text = f.read()
            txt_text.append(text)
    return txt_text, txt_name


def get_bert_json_result(list_cv, cv_names, api=False):
    cv_names = set(cv_names)
    results = dict.fromkeys(cv_names, None)
    for cv, cv_name in zip(list_cv, cv_names):
        list_cv_ = pd.DataFrame.from_dict(cv)
        top_3_score = list_cv_.sort_values(by=['score'], ascending=False).head(3)[
                ['position', 'score']]
        if not api:
            top_3_score_position_only = list_cv_.sort_values(by=['score'], ascending=False).head(3)[
                ['cv_name', 'position', 'score']]
            st.dataframe(top_3_score_position_only)
        top_3_score = top_3_score.to_dict('records')
        score_ = {}
        for score in top_3_score:
            score_[score['position']] = round(score['score'], 2)
        results[cv_name] = score_
    return results


def get_doc2vec_model(model_dir):
    model_jd = gensim.models.Doc2Vec.load(model_dir + 'model_jd_doc2vec')
    model_resume = gensim.models.Doc2Vec.load(model_dir + 'model_resume_doc2vec')
    return model_jd, model_resume


def get_doc2vec_vectors(model, list_of_text):
    list_of_vector = []
    for text in list_of_text:
        vector = text.split()
        vector = model.infer_vector(vector).reshape(1, -1)
        list_of_vector.append(vector)
    print(len(list_of_vector))
    return list_of_vector


def get_similarity_doc2vec_vectors(vector1, list_of_vector2, list_of_jd_name, api=False):
    matchPercentage = []
    result = {}
    for i in range(len(list_of_vector2)):
        cosine_score = cosine_similarity(list_of_vector2[i], vector1)[0][0]
        matchPercentage.append(round(cosine_score, 2))  # round to two decimal

    top_match = sorted(enumerate(matchPercentage), reverse=True, key=lambda x: x[1])[:3]

    for i in range(len(top_match)):
        if not api:
            st.success(list_of_jd_name[top_match[i][0]] + ": " + str(top_match[i][1]))
        result[list_of_jd_name[top_match[i][0]]] = str(top_match[i][1])
    return result


def get_tcn_model_result(cv_text, max_len, tcn_model, tokenizer):
    result = []
    for text in cv_text:
        text = text.split()
        test_sequences = tokenizer.texts_to_sequences(text)
        test_sequences_padded = pad_sequences(test_sequences, maxlen=max_len,
                                              padding='post', truncating='post')
        predict = tcn_model.predict(test_sequences_padded, verbose=0)
        probs = np.argmax(predict, axis=1)
        knowledge = ''
        for i in range(len(text)):
            if probs[i] == 1:
                knowledge += ' ' + text[i]
        result.append(knowledge)
    return result
