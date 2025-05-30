from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import csv
import matplotlib.pyplot as plt
from scipy.sparse import hstack

# The function to retrieve if a token itself contains uppercase
# and store as a column in conll
def extract_uppercase(input_file, output_file):
    outfile = open(output_file, 'w')
    infile = open(input_file,'r')
    for line in infile:
        components = line.rstrip('\n').split()
        if len(components) > 0:
            token = components[0]
            contain_uppercase = any(char.isupper() for char in token)
            if contain_uppercase:
                outfile.write(line.rstrip('\n') + '\t' + '1' + '\n')
            else:
                outfile.write(line.rstrip('\n') + '\t' + '0' + '\n')

    outfile.close()

# The function to retrieve if the previous token contains uppercase
# and store as a column in conll
def extract_previous_case(input_file, output_file):
    token_uppercase = []
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            components = line.rstrip('\n').split()
            if len(components) > 0:              
                uppercase =  components[4]
                token_uppercase.append(uppercase)

    with open(output_file, 'w') as outfile:
        for i, line in enumerate(lines):
            if i == 0:
                outfile.write(line.rstrip('\n') + '\t' + '0' + '\n')
            elif token_uppercase[i-1] == '1':
                outfile.write(line.rstrip('\n') + '\t' + '1' + '\n')
            elif token_uppercase[i-1] == '0':
                outfile.write(line.rstrip('\n') + '\t' + '0' + '\n')
        
    outfile.close()

# The function to retrieve if the next token contains uppercase
# and store as a column in conll
def extract_next_case(input_file, output_file):
    token_uppercase = []
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            components = line.rstrip('\n').split()
            if len(components) > 0:              
                uppercase =  components[4]
                token_uppercase.append(uppercase)

    with open(output_file, 'w') as outfile:
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                outfile.write(line.rstrip('\n') + '\t' + '0' + '\n')
            elif token_uppercase[i+1] == '1':
                outfile.write(line.rstrip('\n') + '\t' + '1' + '\n')
            elif token_uppercase[i+1] == '0':
                outfile.write(line.rstrip('\n') + '\t' + '0' + '\n')
    
        
    outfile.close()

# Extracting the previous token of the current token as a feature
def extract_previous_token(input_file, output_file):
    token_list = []
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            components = line.rstrip('\n').split()
            if len(components) > 0:              
                token =  components[0]
                token_list.append(token)

    with open(output_file, 'w') as outfile:
        for i, line in enumerate(lines):
            if i == 0:
                outfile.write(line.rstrip('\n') + '\t' + '[first]' + '\n')
            else:
                outfile.write(line.rstrip('\n') + '\t' + token_list[i-1] + '\n')
        
    outfile.close()

# Extracting the next token of the current token as a feature
def extract_next_token(input_file, output_file):
    token_list = []
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            components = line.rstrip('\n').split()
            if len(components) > 0:              
                token =  components[0]
                token_list.append(token)

    with open(output_file, 'w') as outfile:
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                outfile.write(line.rstrip('\n') + '\t' + '[last]' + '\n')
            else:
                outfile.write(line.rstrip('\n') + '\t' + token_list[i+1] + '\n')
        
    outfile.close()

# The function to retrieve the prefix and suffix with a span of 6, imiataing the setting
# of the feature discussion paper and store as a column in conll
def extract_affix(input_file, output_file, prefix_length = 6, suffix_length = 6):
    outfile = open(output_file, 'w')
    infile = open(input_file, 'r')
    for line in infile:
        components = line.rstrip('\n').split()
        if len(components) > 0:
            token = components[0]
            prefix = token[:prefix_length] if len(token) >= prefix_length else token
            suffix = token[-suffix_length:] if len(token) >= suffix_length else token
            outfile.write(line.rstrip('\n') + '\t' + prefix + '\t' + suffix + '\n')

    outfile.close()

# Sort steps above into a pipeline for convenience
def feature_pipeline(input_file, output_file):
    extract_uppercase(input_file, f'./{input_file}_1.conll')
    extract_previous_case(f'./{input_file}_1.conll', f'./{input_file}_2.conll')
    extract_next_case(f'./{input_file}_2.conll', f'./{input_file}_3.conll')
    extract_previous_token(f'./{input_file}_3.conll', f'./{input_file}_4.conll')
    extract_next_token(f'./{input_file}_4.conll', f'./{input_file}_5.conll')
    extract_affix(f'./{input_file}_5.conll', output_file)


# Define a new extraction function to sort all current one-hot features into
# a dictionary and also get the gold labels of the specified data
def extract_features_and_labels_onehot(trainingfile):
    """
    The specific function extracts both word tokens and their gold NE annotations
    from the .conll file of the training data. The extraction of word tokens is 
    identical to the function 'extract_features'; the extraction of gold NE annotations
    is achieved by retrieving the last column of every line and storing them in the 
    'target' list. The function eventually returns both the 'data' list and the 'target'
    list.

    :params trainingfile: the .conll file of the training data.
    """
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                pos_tag = components[1]
                syntactic_tag = components[2]
                uppercase = components[4]
                prev_uppercase = components[5]
                next_uppercase = components[6]
                prev_token = components[7]
                next_token = components[8]
                prefix = components[9]
                suffix = components[10]
                feature_dict = {'token':token, 'pos_tag':pos_tag, 'syntactic_tag':syntactic_tag, 'uppercase': uppercase,
                               'prev_uppercase': prev_uppercase, 'next_uppercase':next_uppercase, 'prev_token': prev_token, 
                                'next_token': next_token, 'prefix': prefix, 'suffix': suffix}
                data.append(feature_dict)
                # gold is the 3rd column
                targets.append(components[3])
    return data, targets
    
def extract_features_onehot(inputfile):
    """
    The specific function extracts both word tokens and their designated features
    from the .conll file of the training data. The extraction of word tokens is 
    identical to the function 'extract_features'; the extraction of gold NE annotations
    is achieved by retrieving the last column of every line and storing them in the 
    'target' list. The function eventually returns both the 'data' list and the 'target'
    list.

    :params trainingfile: the .conll file of the training data.
    """
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                pos_tag = components[1]
                syntactic_tag = components[2]
                uppercase = components[4]
                prev_uppercase = components[5]
                next_uppercase = components[6]
                prev_token = components[7]
                next_token = components[8]
                prefix = components[9]
                suffix = components[10]
                feature_dict = {'token':token, 'pos_tag':pos_tag, 'syntactic_tag':syntactic_tag, 'uppercase': uppercase,
                               'prev_uppercase': prev_uppercase, 'next_uppercase':next_uppercase, 'prev_token': prev_token, 
                                'next_token': next_token, 'prefix': prefix, 'suffix': suffix}
                data.append(feature_dict)
    return data
    
def create_classifier(train_features, train_targets, modelname):
    if modelname ==  'logreg':
        # TIP: you may need to solve this: https://stackoverflow.com/questions/61814494/what-is-this-warning-convergencewarning-lbfgs-failed-to-converge-status-1
        model = LogisticRegression(max_iter = 1000)
    elif modelname == 'NB':
        model = MultinomialNB()
    elif modelname == 'SVM':
        model = LinearSVC(C=1.0, max_iter = 1000)
        
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    model.fit(features_vectorized, train_targets)
    
    return model, vec
    
    
def classify_data(model, vec, inputdata, outputfile):
  
    features= extract_features_onehot(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()

def extract_embeddings_as_features_and_gold(trainingfile, word_embedding_model):
    """
    Extract both one-hot features and word embeddings for tokens (and their neighbours) in the training data.
    :param trainingfile: the .conll file of the training data.
    :param word_embedding_model: a pretrained word embedding model.
    :type trainingfile: string
    :type word_embedding_model: string of the model file name for KeyedVectors to load
    :return data: list of dictionaries containing features (with word embeddings replacing the 'token' field)
    :return targets: list of gold labels
    """
    data = []
    targets = []
    
    # Load the pre-trained word embedding model
    model = KeyedVectors.load_word2vec_format(f'./model/{word_embedding_model}', binary=True)
    
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                pos_tag = components[1]
                syntactic_tag = components[2]
                uppercase = components[4]
                prev_uppercase = components[5]
                next_uppercase = components[6]
                prev_token = components[7]
                next_token = components[8]
                prefix = components[9]
                suffix = components[10]
                
                # Try to get the word embedding for the token
                if token in model:
                    token_embedding = model[token].tolist()  # Convert numpy array to list for compatibility
                else:
                    token_embedding = [0] * model.vector_size  # Use a zero vector if token is not in the embedding model

                # Try to get the word embedding for the prev_token
                if prev_token in model:
                    prev_token_embedding = model[prev_token].tolist()  # Convert numpy array to list for compatibility
                else:
                    prev_token_embedding = [0] * model.vector_size  # Use a zero vector if token is not in the embedding model

                # Try to get the word embedding for the prev_token
                if next_token in model:
                    next_token_embedding = model[next_token].tolist()  # Convert numpy array to list for compatibility
                else:
                    next_token_embedding = [0] * model.vector_size  # Use a zero vector if token is not in the embedding model
                    
                
                # Create the feature dictionary with token embedding replacing the 'token' field
                feature_dict = {
                    'token_embedding': token_embedding,  # Replace 'token' with 'token_embedding'
                    'prev_token_embedding': prev_token_embedding, # Replace 'prev_token' with 'prev_token_embedding'
                    'next_token_embedding': next_token_embedding, # Replace 'next_token' with 'next_token_embedding'
                    'pos_tag': pos_tag,
                    'syntactic_tag': syntactic_tag,
                    'uppercase': uppercase,
                    'prev_uppercase': prev_uppercase,
                    'next_uppercase': next_uppercase,
                    'prefix': prefix,
                    'suffix': suffix
                }
                data.append(feature_dict)
                targets.append(components[3])  # Gold label in the 3rd column


    return data, targets

def prepare_combined_features(dict_features):
    """
    Convert feature dictionaries and targets into SVM-compatible matrices.
    :param dict_features: List of feature dictionaries
    :return: feature_matrix (NumPy array), dict_vectorizer (one-hot encoding)
    """
    # Extract embeddings for token, prev_token, and next_token
    embeddings = np.array([
        np.concatenate([
            sample.pop('token_embedding'),
            sample.pop('prev_token_embedding'),
            sample.pop('next_token_embedding')
        ])
        for sample in dict_features
    ])  # Combine token, prev_token, and next_token embeddings for each sample
    dict_vectorizer = DictVectorizer()  # To handle one-hot features
    one_hot_features = dict_vectorizer.fit_transform(dict_features)  # Convert remaining dictionary entries to one-hot vectors
    
    # Combine embeddings and one-hot features
    feature_matrix = hstack([embeddings, one_hot_features])  # Horizontally stack embedding and one-hot, using the hstack from scipy to avoid error
    
    
    return feature_matrix, dict_vectorizer

def prepare_combined_features_for_input(dict_features, dict_vectorizer):
    """
    Convert new data dictionaries into feature matrices using a pre-trained DictVectorizer.
    :param dict_features: List of feature dictionaries
    :param dict_vectorizer: DictVectorizer from training
    :return: feature_matrix
    """
    # Extract embeddings for token, prev_token, and next_token
    embeddings = np.array([
        np.concatenate([
            sample.pop('token_embedding'),
            sample.pop('prev_token_embedding'),
            sample.pop('next_token_embedding')
        ])
        for sample in dict_features
    ])  # Combine token, prev_token, and next_token embeddings for each sample
    one_hot_features = dict_vectorizer.transform(dict_features)  # Use the DictVectorizer from training
    feature_matrix = hstack([embeddings, one_hot_features])  # Combine embeddings and one-hot features using scipy hstack to avoid error
    return feature_matrix

def create_classifier_with_w2v(train_features, train_targets, modelname):
    if modelname ==  'logreg':
        # TIP: you may need to solve this: https://stackoverflow.com/questions/61814494/what-is-this-warning-convergencewarning-lbfgs-failed-to-converge-status-1
        model = LogisticRegression()
    elif modelname == 'NB':
        model = MultinomialNB()
    elif modelname == 'SVM':
        model = LinearSVC(C=0.1, max_iter=5000)

    feature_matrix_with_w2v, vec_with_w2v = prepare_combined_features(train_features)
    model.fit(feature_matrix_with_w2v, train_targets)
    
    return model, vec_with_w2v

def classify_data_w2v(embedding, classifier, vec, inputdata, outputfile):
    features, _ = extract_embeddings_as_features_and_gold(inputdata, embedding)
    combined_features = prepare_combined_features_for_input(features, vec)
    predictions = classifier.predict(combined_features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()

#----------------------------------------------------------------------------------------------
# Evaluation Functions
def retrieve_metrics_and_cm(gold_file, pred_file, output_txt, output_image):
    """
    The function to produce metrics report based on a file storing
    gold NER labels and a file storing the NER labels predicted by
    a classifier and save results to the local folder. Both input files should be in .conll format.

    :params gold_file: the .conll file storing gold NER labels
    :params pred_file: the .conll file storing predicted NER labels
    :params output_txt: the text file recording metrics
    :params output_image: the image of the confusion matrix
    
    """
    gold_list = []
    pred_list = []

    with open(gold_file, "r") as infile_gold:
        for line in infile_gold:
            components_g = line.rstrip('\n').split()
            if len(components_g) > 0:
                gold_list.append(components_g[3])
                
    with open(pred_file, "r") as infile_pred:
        for line in infile_pred:
            components_p = line.rstrip('\n').split()
            if len(components_p) > 0:
                #pred_list.append(components_p[3])
                pred_list.append(components_p[-1]) 
                
    report = classification_report(gold_list, pred_list, digits = 3)

    labels = sorted(set(gold_list+pred_list))
    matrix = confusion_matrix(gold_list, pred_list)
    display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = labels)
    
    # Save metrics report to a text file
    with open(output_txt, "w") as outfile:
        outfile.write('Classification Metrics Report\n')
        outfile.write(f'The gold is retrieved from: {gold_file}.\n')
        outfile.write(f'The predictions are retrieved from: {pred_file}.\n')
        outfile.write(report)
        outfile.write('\nConfusion Matrix\n')
        outfile.write(str(matrix))
        outfile.write('\nLabels\n')
        outfile.write(str(labels))
    
    print(f"Metrics report saved to {output_txt}")
    
    # Save confusion matrix as an image
    display.plot()
    plt.savefig(output_image)
    print(f"Confusion matrix image saved to {output_image}")

def retrieve_weighted_metrics(gold_file, pred_file, output_txt):
    """
    A function to retrieve metrics and the confusion matrix based on a weighted scoring system
    (less error weight assigned to partial match by length or entity type)
    
    :params gold_file: the .conll file storing gold NER labels
    :params pred_file: the .conll file storing predicted NER labels
    :params output_txt: the text file recording metrics
    
    """
    # Step 1: Prepare gold and pred lists from files (similar to the original function)
    gold_list = []
    pred_list = []

    with open(gold_file, "r") as infile_gold:
        for line in infile_gold:
            components_g = line.rstrip('\n').split()
            if len(components_g) > 0:
                gold_list.append(components_g[3])
                
    with open(pred_file, "r") as infile_pred:
        for line in infile_pred:
            components_p = line.rstrip('\n').split()
            if len(components_p) > 0:
                #pred_list.append(components_p[3])
                pred_list.append(components_p[-1])

    # Step 2: Create confusion matrix and labels
    labels = sorted(set(gold_list + pred_list))
    cm = confusion_matrix(gold_list, pred_list, labels=labels)

    # Step 3: Define the weight matrix (customize based on needs)
    weight_matrix = np.ones(cm.shape)  # Default weight of 1 for all
    for i, label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            # The order of checking matters here!
            if label == pred_label:
                weight_matrix[i][j] = 1  # standard weight for correct predictions
            elif label == 'O' or pred_label == 'O':
                weight_matrix[i][j] = 1  # standard weight for missed entities (false negatives) 
                                         # or wrongly detected entities false positive
            else:
                weight_matrix[i][j] = 0.5 # all other errors that are not a complete error in extracting entity receive lower weight
            

    # Step 4: Calculate the confusion matrix
    weighted_cm = cm * weight_matrix  # Apply weights to the original confusion matrix

    # Step 5: Calculate precision, recall, f1 for each label based on the weighted confusion matrix
    report = {}
    total_support = 0
    total_precision, total_recall, total_f1, weighted_sum_precision, weighted_sum_recall, weighted_sum_f1 = 0, 0, 0, 0, 0, 0
    
    for i, label in enumerate(labels):
        tp = weighted_cm[i, i]  # True positives for label
        fp = weighted_cm[:, i].sum() - tp  # False positives for label (all predicted positives - tp)
        fn = weighted_cm[i, :].sum() - tp  # False negatives for label (all gold pisutuves - tp)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        support = int(cm[i, :].sum())  # Original support count from unweighted cm
        total_support += support

        # Accumulate for macro averaging
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        # Accumulate for weighted averaging
        weighted_sum_precision += precision * support
        weighted_sum_recall += recall * support
        weighted_sum_f1 += f1 * support
        
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(cm[i, :].sum())  # Original support count from unweighted cm
        }

    # Calculate macro and weighted averages
    num_labels = len(labels)
    macro_avg = {
        "precision": total_precision / num_labels,
        "recall": total_recall / num_labels,
        "f1-score": total_f1 / num_labels,
    }
    weighted_avg = {
        "precision": weighted_sum_precision / total_support,
        "recall": weighted_sum_recall / total_support,
        "f1-score": weighted_sum_f1 / total_support,
    }

    
    # Display the weighted classification report
    with open(output_txt, "w") as outfile:
        outfile.write("Weighted Classification Report:\n")
        for label, metrics in report.items():
            outfile.write(f"{label}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
                          f"F1-Score={metrics['f1-score']:.3f}, Support={metrics['support']}\n")
        
        outfile.write("\nMacro Average:\n")
        outfile.write(f"Precision={macro_avg['precision']:.3f}, Recall={macro_avg['recall']:.3f}, F1-Score={macro_avg['f1-score']:.3f}\n")
        
        outfile.write("\nWeighted Average:\n")
        outfile.write(f"Precision={weighted_avg['precision']:.3f}, Recall={weighted_avg['recall']:.3f}, F1-Score={weighted_avg['f1-score']:.3f}\n")
        
    print(f"Metrics report saved to {output_txt}")

def convert_to_spans(tag_list):
    """
    A function to turn token-level NER tag lists into span tuples in the form of (start, end, type)

    :params tag_list: a token-level NER tag list
    """
    # Initial a span list, two variables used as checking conditions in progress
    spans = []
    start = None
    current_type = None

    # Loop over the enumerated tag list
    for i, tag in enumerate(tag_list):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1, current_type))  # This section is to save the previous NE span if there are two neighbouring NE's, i-1 is where the previous span ends
            start = i  # Write the start with the current index where the new NE starts
            current_type = tag[2:]  # Save the current type
        elif tag.startswith("I-") and current_type == tag[2:]:
            continue  # If this condition is satisfied, it means that the span should continue
        else: # If this condition is satisfied, "O" is met and the span should end
            if start is not None:
                spans.append((start, i - 1, current_type))  # save the span and initialize "start" and "current type" again
            start = None
            current_type = None

    # If there is a non-ended span at the end of the list, save it
    if start is not None:
        spans.append((start, len(tag_list) - 1, current_type))
    return spans

def retrieve_tag_lists(gold_file, pred_file):
    """

    :params gold_file: the .conll file storing gold NER labels
    :params pred_file: the .conll file storing predicted NER labels
    
    """
    # Step 1: Prepare gold and pred lists from files (similar to the original function)
    gold_list = []
    pred_list = []

    with open(gold_file, "r") as infile_gold:
        for line in infile_gold:
            components_g = line.rstrip('\n').split()
            if len(components_g) > 0:
                gold_list.append(components_g[3])
                
    with open(pred_file, "r") as infile_pred:
        for line in infile_pred:
            components_p = line.rstrip('\n').split()
            if len(components_p) > 0:
                #pred_list.append(components_p[3])
                pred_list.append(components_p[-1])
    return gold_list, pred_list

def evaluate_span_based(ground_truth, predictions, output_txt):
    """
    Evaluate span-based NER performance with partial match support and save the metrics
    to a local folder.

    :params ground_truth: a list of ground truth spans in the form (start, end, type)
    :params predictions: a list of predicted spans in the form (start, end, type)
    
    :prints: precision, recall, f1 score
    """
    exact_matches = 0
    partial_matches = 0
    predicted_total = len(predictions)
    ground_truth_total = len(ground_truth)

    # Convert spans to sets for easier comparison
    gt_spans = set(ground_truth)
    pred_spans = set(predictions)

    # Calculate exact matches
    exact_matches = len(gt_spans & pred_spans) # & for intersection, this is possible because each tuple is unique
    
    # Calculate partial matches
    gt_spans -= gt_spans & pred_spans  # Remove exact matches from ground truth
    pred_spans -= pred_spans & gt_spans  # Remove exact matches from predictions

    for gt_start, gt_end, gt_type in gt_spans:
        for pred_start, pred_end, pred_type in pred_spans:
            # scope over spans from two lists partially overlapped on the index (ignore type mismatch here since the focus is on general recognition)
            if not (pred_end < gt_start or pred_start > gt_end):
                # Calculate the overlap ratio based on length for partial matches and use it as the partial score added to stats
                # locating overlap through indices
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)
                overlap_length = overlap_end - overlap_start + 1
                gt_length = gt_end - gt_start + 1
                pred_length = pred_end - pred_start + 1
                overlap_ratio = overlap_length / max(gt_length, pred_length)
                
                # Consider as partial match if there's positive overlap ratio
                if overlap_ratio > 0:
                    partial_matches += overlap_ratio  # Use overlap ratio as partial score
                    pred_spans.remove((pred_start, pred_end, pred_type)) # After adding the ratio, remove the calculated tuple from list to avoid repeated calculation
                    break

    # Calculate precision, recall, and F1 score
    matched_total = exact_matches + partial_matches
    precision = matched_total / predicted_total if predicted_total > 0 else 0
    recall = matched_total / ground_truth_total if ground_truth_total > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    with open(output_txt, "w") as outfile:
        outfile.write(f'precision:{precision}\nrecall:{recall}\nf1_score:{f1_score}.')
    print(f"Metrics report saved to {output_txt}")

    

def span_based_eva_pipeline(gold_file, pred_file,output_txt):
    gold_list, pred_list = retrieve_tag_lists(gold_file, pred_file)
    gold_spans = convert_to_spans(gold_list)
    pred_spans = convert_to_spans(pred_list)
    evaluate_span_based(gold_spans,pred_spans,output_txt)


def main(argv=None):
    
    #a very basic way for picking up commandline arguments
    if argv is None:
        argv = sys.argv
        
    #Note 1: argv[0] is the name of the python program if you run your program as: python program1.py arg1 arg2 arg3
    #Note 2: sys.argv is simple, but gets messy if you need it for anything else than basic scenarios with few arguments
    #you'll want to move to something better. e.g. argparse (easy to find online)
    
    
    #you can replace the values for these with paths to the appropriate files for now, e.g. by specifying values in argv
    #argv = ['mypython_program','','','']
    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
    w2v_model_name = argv[4]

    processed_trainingfile = './final_train_onehot_features.conll'
    processed_inputfile = './final_input_onehot_features.conll'
    
    feature_pipeline(trainingfile, processed_trainingfile)
    feature_pipeline(inputfile, processed_inputfile)
    
    training_features, gold_labels = extract_features_and_labels_onehot(processed_trainingfile)
    for modelname in ['logreg', 'NB', 'SVM']:
        ml_model, vec = create_classifier(training_features, gold_labels, modelname)
        classify_data(ml_model, vec, processed_inputfile, outputfile.replace(".conll", f".{modelname}.conll"))
        retrieve_metrics_and_cm(inputfile, outputfile.replace(".conll", f".{modelname}.conll"), f"{outputfile.replace('.conll', f'.{modelname}.txt')}", f"{outputfile.replace('.conll', f'.{modelname}.png')}")
        #retrieve_weighted_metrics(inputfile, outputfile.replace(".conll", f".{modelname}.conll"), f"{outputfile.replace('.conll', f'.{modelname}_weighted.txt')}")
        span_based_eva_pipeline(inputfile, outputfile.replace(".conll", f".{modelname}.conll"), f"{outputfile.replace('.conll', f'.{modelname}_span.txt')}")

    w2v_train_features, w2v_gold = extract_embeddings_as_features_and_gold(processed_trainingfile, w2v_model_name)
    w2v_svm_classifier, w2v_svm_vec = create_classifier_with_w2v(w2v_train_features, w2v_gold, 'SVM')
    classify_data_w2v(w2v_model_name, w2v_svm_classifier, w2v_svm_vec, processed_inputfile, outputfile.replace(".conll", ".w2v_svm.conll"))
    retrieve_metrics_and_cm(inputfile, outputfile.replace(".conll", ".w2v_svm.conll"), f"{outputfile.replace('.conll', '.w2v_svm.txt')}", f"{outputfile.replace('.conll', '.w2v_svm.png')}")
    #retrieve_weighted_metrics(inputfile, outputfile.replace(".conll", ".w2v_svm.conll"), f"{outputfile.replace('.conll', '.w2v_svm_weighted.txt')}")
    span_based_eva_pipeline(inputfile, outputfile.replace(".conll", ".w2v_svm.conll"), f"{outputfile.replace('.conll', '.w2v_svm_span.txt')}")
    
if __name__ == '__main__':
    main()