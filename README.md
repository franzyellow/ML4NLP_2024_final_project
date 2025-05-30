## ML4NLP 2024 Final Project - Machine Learning for Named Entity Recognition and Classification
1. The main code file: [ner_machine_learning.py](ner_machine_learning.py)
   - The "GoogleNews-vectors-negative300.bin" file needs to be put under the [model]() folder in the same path as the main code file.
   - Then redirect the terminal to the file path of the main code file, and execute the following command:  
     `python ner_machine_learning.py conll2003.train.conll conll2003.test.conll pred_test.conll GoogleNews-vectors-negative300.bin`  
     It will produce all the intermediate files and result metrics in the same path.
2. [Feature Ablations & Error Analysis.ipynb](Feature%20Ablations%20&%20Error%20Analysis.ipynb)
   - These two modules are stored separately in a notebook as they contain plot visualisations and it seems reasonable to be separated from the main codes for executing efficiency.
   - After executing [ner_machine_learning.py](ner_machine_learning.py), one can "run all blocks" for this notebook directly to reproduce the process. The produced file will be stored in the [ablations]() folder.
3. [Hyperparameter tuning & experimental codes.ipynb](Hyperparameter%20tuning%20&%20experimental%20codes.ipynb)
   - The notebook contains the hyperparameter tuning for the SVM-wmb model at the bottom. Other parts are experimental codes which eventually led to [ner_machine_learning.py](ner_machine_learning.py)
4. Folders other than "ablations" and "model":
   - Storing all the metric files, confusion matrices, figures and intermediate .conll files that I used for the report writing. Attached for reference.
