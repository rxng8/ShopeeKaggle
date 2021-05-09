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

<details>
  <summary> <h1>Other works</h1> </summary>

  ## Week 1: Feb 1 - Feb 5: [Chest X-ray project week 1](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-1-feb-1---feb-5)

  ## Week 2: Feb 8 - Feb 12: [Chest X-ray project week 2](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-2-feb-8---feb-12)

  ## Week 3: Feb 15 - Feb 19: [Chest X-ray project week 3](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-3-feb-15---feb-19)

  ## Week 4: Feb 22 - Feb 26: [Chest X-ray project week 4](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-4-feb-22---feb-26)

  ## Week 5: March 1 - March 5: [Chest X-ray project week 5](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-5-march-1---march-5)

  ## Week 6: March 8 - March 12: [Chest X-ray project week 6](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-6-march-8---march-12)

  ## Week 7: March 15 - March 19: [Chest X-ray project week 7](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-7-march-15---march-19)

  ## Week 8: March 22 - March 26: [Chest X-ray project week 8](https://github.com/rxng8/Chest-Xray-Abnormalities-Detection#week-8-march-22---march-26)
  
</details>

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

## Week 12: April 18 - April 22:
* [**2 hours**] Read the paper [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) and try to go through the algorithms.
* [**0.5 hours**] Write utilities function that can be seen in `utils.py`:
  * `preprocess()` method to use in preprocessing the test set.
  * `rgb2gray()` converts the 3-channel image to grayscale image.
  * `ConvNet` tensorflow model class which evaluate the images.
  * Other supporting layer methods such as `down_conv()`, `dropout_layer()`, etc.
* [**4.5 hours**] Examine the ways to convert words to vectors:
  * [**1.5 hours**] Read, research, and Implement TF-IDF Vectorizers in file `vectorizers.py`
  * [**1 hour**] Read of about bags of words.
  * [**1.5 hours**] Read about how to use the Embedding layer in Deep learning to embed the text to vector with machine.
  * [**0.5 hours**] Write test template for classes of word vectorizers in file `word2vec.py`
* [**0.5 hours**] Test FAISS with vectors and visualize in `faiss_test.py`
* [**1.5 hours**] Work on the FAISS Application in the actual program `notebook.py`, fitted data vectors from tfidf vectorizers. Plan to use text embedding next week.

<details>
  <summary> <h1>Other works</h1> </summary>

  ## Week 13: April 25 - April 28: [Faster R-CNN Research Week 13](https://github.com/rxng8/Faster-R-CNN-Research#week-13-april-25---april-28)
  
  ## Week 14: May 1 - May 5: [Faster R-CNN Research Week 13](https://github.com/rxng8/Faster-R-CNN-Research#week-14-may-2---may-6)

</details>

-----------------

