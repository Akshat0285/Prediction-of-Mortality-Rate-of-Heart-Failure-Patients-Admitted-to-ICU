# from classification2 import classification
from classification3 import Classification

import warnings
warnings.filterwarnings("ignore")

clf=Classification(clf_opt='ab', impute_opt='knn')
clf.get_data()

clf.classification()
