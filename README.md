# Shopee's Categorizing Listings Kaggle Competition

## About the project:
This project is created for academic purpose and also available publicly. More information about the competition can be found [here](https://www.kaggle.com/c/shopee-product-matching).

# Prerequisites
* [Anaconda](https://www.anaconda.com/products/individual)
* Kaggle API:
  * [Set up kaggle](https://www.kaggle.com/docs/api). 
  * [Create api key and move the key file to ./kaggle folder.](https://github.com/Kaggle/kaggle-api/issues/15#issuecomment-500713264).
* If you are using window, you will have to use window's linux subsystem to run the bash script which download the dataset. Or you can download directly from kaggle.
* Install the prerequisite library:
  ```
  pip install -r requirements.txt
  # or
  conda env create -f environment.yml
  ```
I recommend you use conda virtual environment and use the conda command instead of pip

# Run the project
Go to [`notebook.py`](./notebook.py) to evaluate each cell.

---------------
# Project Backlog:

## Week 9: March 29 - April 2:
* **[3 hours]** Analyzing the nature of the data.
  * **[2 hours]** Analyzing the models should be used: TFIDF and ResNEt Transfer learning.
  * **[1 hours]** Analyzing the columns of the training dataset.

* **[6 hours]** Wrting the [`dataset`](./core/data.py) class
  * **[1 hours]** Set up conda environment
  * **[3 hours]** Creating the whole pipeline
  *  **[1 hours]** Writing pandas dataframe data analysis (groupby, concatenation)
  *  **[1 hours]** Wrting [`notebook.py`](./notebook.py).

## Week 10: April 5 - April 9:
* [**1 hours**] Set up virtual environment. Look at the file [`requirements.txt`](./requirements.txt) and [`environment.yml`](./environment.yml):
  * Add tensorflow library.
  * Add open cv library.
  * Add pytorch library.
  * Add other realted library: tensorboard, tfmodel, etc.
  * Update documentation about the environment requirements and installation.
* [**2 hours**]
  * Research about TF-IDF model, code it in the [`notebook`](./notebook.py)
* [**3 hours**] Reasearch about the Facebook's FAISS text to vector store. For the use of finding nearest text vector instead of using neighboring algorithms.
* [**2 hours**] Research about how to code K-nearest neighbor in the context of the text. Train the K-nearest neighbor model on the dataset of over 65,000 instances.
* [**1 hours**] Finish the notebook pipeline using sklearn without parameters tuned and create submission.

## Week 11: April 12 - April 15:
* [**1 hours**] Write [`utils`](./core/utils.py) class for image preprocessing, ram, cpu, and gpu control when training pytorch model. Install libraries and update [`environment.yml`](environment.yml) and [`requirements.txt`](requirements.txt).
* [**4 hours**] Write pytorch transfer learning pipeline in [`notebook.py`](./notebook.py).
  * Pytorch dataset
  * Image transpose
  * Initialize model method which dynamically initialize model with names.
* [**2 hours**] Read and research about FAISS algorithms from Facebook:
  * Reference: [https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
  * Run test on 1-million-random-vectors with dimension 50.
  * This algorithm is very fast for vector searching.
* [**2 hours**] Train and export KNN numpy vector model (distances, indices). Prepare work for next week's prediction.

---------------

# Tentative:

## Week 12: April 18 - April 22:

## Week 13: April 25 - April 28:

## Week 14: May 1 - May 5:

## Week 15: May 8 - May 12:

## Week 16: May 15 - May 19:

-----------------

