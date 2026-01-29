**Detecting Spam Emails Using Tensorflow in Python**

Spam messages are unsolicited or unwanted emails/messages sent in bulk to users. Detecting spam emails automatically helps prevent unnecessary clutter in users' inboxes.

In this article, we will build a spam email detection model that classifies emails as Spam or Ham (Not Spam) using TensorFlow, one of the most popular deep learning libraries.

**Step 1: Import Required Libraries**
Before we begin let’s import the necessary libraries: pandas, numpy, tensorflow, matplotlib, wordcloud, nltk for data processing, model building, and visualization.

**Step 2: Load the Dataset**
We’ll use a dataset containing labeled emails (Spam or Ham). Let’s load the dataset and inspect its structure.

**Step 3: Balance the Dataset**
We can clearly see that number of samples of Ham is much more than that of Spam which implies that the dataset we are using is imbalanced. To address the imbalance we’ll downsample the majority class (Ham) to match the minority class (Spam).

**Step 4: Clean the Text**
Textual data often requires preprocessing before feeding it into a machine learning model. Common steps include removing stopwords, punctuations, and performing stemming/lemmatization.

We’ll perform the following steps:

1. Stopwords Removal
2. Punctuations Removal
3. Stemming or Lemmatization

**Step 5: Tokenization and Padding**
Machine learning models work with numbers, so we need to convert the text data into numerical vectors using Tokenization and Padding.

1. Tokenization: Converts each word into a unique integer.
2. Padding: Ensures that all text sequences have the same length, making them compatible with the model.

**Step 6: Define the Model**
We will build a deep learning model using a Sequential architecture. This model will include:

1.Embedding Layer: Learns vector representations of words.
2. LSTM Layer: Captures patterns in sequences.
3. Fully Connected Layer: Extracts relevant features.
4. Output Layer: Predicts whether an email is spam or not.

**Step 7: Train the Model**
We train the model using EarlyStopping and ReduceLROnPlateau callbacks. These callbacks help stop the training early if the model’s performance doesn’t improve and reduce the learning rate to fine-tune the model.
