# Description
We have built a Resume & Job Description Matching System using Deep Learning.

# Installation

```shell
pip install -q requirements.txt
```

# Demo

Step 1 : Prepare the Resumes and JDs. You can get the JDs [here](https://drive.google.com/drive/folders/12YDeelF66Qg6Im7aRHNq6C-ef1oGiZY5?usp=sharing). Store JDs with path "./data/preprocessed_txt_JD/" 

Step 2 : Download the model from [link](https://drive.google.com/drive/folders/1pbTHnXARHGSiMDWbSwrYkDyhByOuVBM7?usp=sharing). Store it with path "./data/model/"

Step 3 : Run demo with Streamlit

```shell 
streamlit run test.py
```

The demo app will be opened at http://localhost:8501/

Step 4 (Optional): FastAPI 

```shell
python app.py
```

You can view the docs at http://localhost:8008/docs

# Usage/Example
Step 1: Choose the resume to upload to system.

![img.png](./image/img.png)

Step 2: Choose to get recommendation with BERT model or Doc2Vec model.

![img2.png](./image/img2.png)

Step 3: Show the results.

![img3.png](./image/img3.png)