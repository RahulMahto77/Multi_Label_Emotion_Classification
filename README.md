
# GoEmotions : text-based emotion detection

## Introduction
Nowadays, virtual interactions occupy an important place in our lives, especially in the COVID-19 context.

We wanted to address the problem of text-based emotion detection as it is really difficult to interpret emotions in a text message or any social media comment. There is always a lot of subjectivity in the way sentences are turned (sarcasm, irony, etc), in addition to the explosion of emojis use.

Some of the applications are: social media analysis, detection of emotional distress, etc.

## Dataset
we used the GoEmotions dataset [1].

This dataset was built by a Google Research team and gathers **more than 58K Reddit english comments**. In fact, it is **the largest manually labeled dataset** in this topic.

However, this dataset presents several challenges:

 - **Very high number of emotions** to detect: 27 emotions + "neutral"
 - **Class imbalance**: ~30% of "neutral" samples
 - **Multi-label**: each sample can be labeled with up to 5 different emotions

### Final_year_project.ipynb
 **Exploration and cleaning of the data in preparation for multi-label text classification tasks**
 
 - Input files
	 - train.tsv: train dataset
	 - dev.tsv: validation dataset
	 - test.tsv: test dataset
	 - emotions.txt: list of emotions in the GoEmotions taxonomy
 - Output files
	 - train_clean.csv: clean train dataset
	 - val_clean.csv: clean validation dataset
	 - test_clean.csv: clean test dataset

### Baseline_Modeling.ipynb
 **Creating baseline models for emotion detection**
 
 - Input files
	 - train_clean.csv: clean train dataset
	 - val_clean.csv: clean validation dataset
	 - test_clean.csv: clean test dataset
	 - emotions.txt: list of emotions in the GoEmotions taxonomy

###  BERT_Model.ipynb
 **Fine-tuning a BERT model for emotion detection (without "neutral" samples) **
 
 - Input files
	 - train_clean.csv: clean train dataset
	 - val_clean.csv: clean validation dataset
	 - test_clean.csv: clean test dataset
	 - emotions.txt: list of emotions in the GoEmotions taxonomy
	 - ekman_labels.txt: list of emotions in the Ekman taxonomy
 - Output files
	 - bert-weights.hdf5: model weights

### my-annoying-shrink-app :) 

It mimics a therapist that answers his patients with irritating replies, only according to the emotions he detected after he asked the question: *"how are you feeling today ?"* 

Each time the patient (you) enters a text, the therapist (our model) analyzes the emotions with their probabilities, and and the app returns these in a web interface together with a predefined "answer" for each emotion.
