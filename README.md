# Active-Feature-Elicitation

This repository contains the code for the paper "On Whom Should I Perform the Lab Test on Next? An Active Feature Elicitation Approach" which has been published in International Joint Conference on AI (IJCAI) 2018.  The link for the paper is http://www.ijcai.org/proceedings/2018/0486.pdf. Folds for Pima dataset has been uploaded as a test data set for this code. For any queries related to the code, drop an email to sxd170431@utdallas.edu.

Run instructions:

Run the code active_feature_acq.py in the below mentioned way.
active_feature_acq.py "D:\git_projects\Active-Feature-Elicitation\Data" Pima Pima_clean GB KL 5 5 14 5 10 5 6 3 0
 
Details of the command line arguments are as below:

1. path (Folder named "Pima" where all files are kept) data set name  
2. Folder named "Pima" where all files are kept
3. Data set name "Pima_clean"
4. Classifier used:- GB: Gradient Boosting, LR: Logistic Regression, SVM: Support Vector Machine, DT: Decision tree. You can add your own classifier in Classifier.py
5. Divergence metric used:- KL: KL-divergence, HD: Hellinger distance, TV: Total Variation, CD: Chi squared distance 
6. Number of observed features (1+actual number of features, the first column in Pima_clean data set contains identifiers which are also included for convenience). The unique identifiers in the 1st column are however not considered while learning the models.
7. Size of data points to be acquired in every iteration of Active learning
8. No. of iterations of data point acquisition in each run
9. No. of runs 
10. Observed set sample size to start with 
11. test-split. 1/test_split gives the size of the test_set. If test-split=5, test_size=0.2 which means 80-20 is the train test split.
12. Depth of the Gradient Boosting classifier on all the features
13. Depth of the Gradient Boosting classifier on the observed features
14. pos_to_neg ratio: This parameter is currently not required in the code, can set it to 0 

Important instructions:

1. Folder structure should be the same as there in the Data Folder.
2. The IJCAI paper has all the experiments using Gradient Boosting Classifier and KL-divergence as the distance metric. Hence, these are currently there in the folder structure.
3. Every Classifier and distance metric has separate folders where results gets stored. In order to try a new classifier, you have to create a new folder with the desired name.
4. The code runs the algorithm and also runs the Random baseline as mentioned in the paper. 
5. In the All_runs folder, Pima_clean_stat.csv will give the performance metric of all the runs. In the same folder, *_sugg.csv will contain individual values of all iterations across all runs. *_rnd.csv will contain details about the random baseline.
6. Graphs folder will contain the graphs.
7. The 5-folds for the experiments have already been created and stored in the Fold* folders.
8. The above mentioned parameter setting is the one that is reported in the paper. The parameters are also mentioned in settings.txt file.
