import pythoncom
from utils import *
from config import *
import torch
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()


def prepare_data():
    preprocessing_text(uploaded_file_path=UPLOADED_FILE_PATH,
                       pdf_path=PDF_PATH,
                       txt_path=TXT_PATH,
                       preprocessed_path=PREPROCESSED_PATH)
    jd_text, jd_name = get_txt_list(PREPROCESSED_JD_PATH)
    cv_text, cv_names = get_txt_list(PREPROCESSED_PATH)
    return jd_text, jd_name, cv_text, cv_names


@app.post("/get_bert_recommendation")
async def get_bert_recommendation(uploaded_files: List[UploadFile] = File(...)):
    """
    Requires an image file upload and Optional image size parameter.
    Intended for API users.
    Return: JSON results of running YOLOv5 on the uploaded image.
    """
    for uploaded_file in uploaded_files:
        save_uploaded_file_fast_api(uploaded_file)
    jd_text, jd_name, cv_text, cv_names = prepare_data()

    # Remove cached files
    if os.path.exists(UPLOADED_FILE_PATH):
        shutil.rmtree(UPLOADED_FILE_PATH)
    if os.path.exists(PDF_PATH):
        shutil.rmtree(PDF_PATH)
    if os.path.exists(TXT_PATH):
        shutil.rmtree(TXT_PATH)
    if os.path.exists(PREPROCESSED_PATH):
        shutil.rmtree(PREPROCESSED_PATH)

    device = "cpu"
    cv_jd_pair = []
    for text, name in zip(cv_text, cv_names):
        cv_jd_pair.append({
            'CV': [text] * len(jd_text),
            'JD': jd_text,
            'score': [],
            'position': jd_name,
            'cv_name': name
        })

    model = SentenceTransformer(model_name_or_path=SBERT_MODEL_PATH, device=device)
    for item in cv_jd_pair:
        item['score'] = cal_cosine_score(model=model,
                                         sentences1=item['CV'],
                                         sentences2=item['JD'],
                                         show_progress_bar=True)
    print_memory_usage()
    BERT_json_results = get_bert_json_result(list_cv=cv_jd_pair, cv_names=cv_names, api=True)
    return BERT_json_results


@app.post("/get_doc2vec_recommendation")
async def get_doc2vec_recommendation(uploaded_files: List[UploadFile] = File(...)):
    for uploaded_file in uploaded_files:
        save_uploaded_file_fast_api(uploaded_file)
    jd_text, jd_name, cv_text, cv_names = prepare_data()

    # Remove cached files
    if os.path.exists(UPLOADED_FILE_PATH):
        shutil.rmtree(UPLOADED_FILE_PATH)
    if os.path.exists(PDF_PATH):
        shutil.rmtree(PDF_PATH)
    if os.path.exists(TXT_PATH):
        shutil.rmtree(TXT_PATH)
    if os.path.exists(PREPROCESSED_PATH):
        shutil.rmtree(PREPROCESSED_PATH)

    # Get tcn_model results
    tcn_model, tokenizer = get_tcn_model(model_dir=TCN_MODEL_PATH, tokenizer_dir=TOKENIZER_PATH)
    tcn_results = get_tcn_model_result(cv_text=cv_text, max_len=MAX_LENGTH_TCN_MODEL,
                                       tcn_model=tcn_model, tokenizer=tokenizer)

    # Get jd_text
    jd_text, jd_name = get_txt_list(TCN_PREPROCESSED_JD_PATH)
    # Get doc2vec model results
    doc2vec_model_jd, doc2vec_model_cv = get_doc2vec_model(model_dir=DOC2VEC_MODEL_PATH)
    jd_vectors = get_doc2vec_vectors(model=doc2vec_model_jd, list_of_text=jd_text)
    cv_vectors = get_doc2vec_vectors(model=doc2vec_model_cv, list_of_text=tcn_results)
    doc2vec_json_results = {}
    for cv_vector, cv_name in zip(cv_vectors, cv_names):
        st.write("Top 3 recommendations for " + cv_name + " is: ")
        result = get_similarity_doc2vec_vectors(vector1=cv_vector, list_of_vector2=jd_vectors,
                                                list_of_jd_name=jd_name, api=True)
        doc2vec_json_results[cv_name] = result
    return doc2vec_json_results


if __name__ == '__main__':
    pythoncom.CoInitialize()
    app_str = 'app:app'
    uvicorn.run(app_str, host='localhost', port=8008, workers=1)
