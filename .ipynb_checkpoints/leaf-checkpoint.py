### Libraries ###

from ipywidgets import IntProgress, Textarea, HTML, HBox, Label
from IPython.display import display
import tabulate
import matplotlib.colors as mc
import colorsys
import warnings, importlib, copy, random, mock, os, math, ast, sys, re, itertools, time, datetime
import numpy as np
import numpy.linalg as linalg
# from numpy import linalg as linalg
import matplotlib.pyplot as plt
from numpy.random import lognormal, normal, exponential
from scipy.stats import multivariate_normal, norm, spearmanr
import pandas as pd
import seaborn as sns
import sklearn, tqdm, scipy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from scipy.spatial.distance import pdist, cdist
from imblearn.over_sampling import SMOTE
from collections import Counter
import scipy.stats
from termcolor import colored
from time import sleep
import shap
from fastprogress.fastprogress import master_bar, progress_bar

import warnings
warnings.filterwarnings("ignore", message='y_pred contains classes not in y_true')




###########################################################################################


#############################################################################################################

def train_model(X, Y, model, verbose=True):
    # separate train and test sets
    train, test, labels_train, labels_test = (
        sklearn.model_selection.train_test_split(X, Y, train_size=0.80, test_size=0.20,
                                                 random_state=1234))

    cls_type = model.__class__.__name__
    use_weights = not (cls_type in ['MLPClassifier', 'KNeighborsClassifier', 'GaussianProcessClassifier'])

    if use_weights:
        nT = sum(labels_train)
        nF = len(labels_train) - nT
        weights_train = ((labels_train==False)*nT + (labels_train==True)*nF) / len(labels_train)
    
        model.fit(train, labels_train, sample_weight=weights_train)
    else:
        model.fit(train, labels_train)
    
    # verify the classifier on the test set
    pred_test = model.predict(test)
    # print(pred_test, type(pred_test), pred_test.dtype)
    # pred_test = pred_test > 0.5 if str(pred_test.dtype).startswith('float') else pred_test
    if use_weights:
        nT = sum(labels_test)
        nF = len(labels_test) - nT
        weights_test = ((labels_test==False)*nT + (labels_test==True)*nF) / len(labels_test)
        print('  *', cls_type, 'accuracy:', 
              sklearn.metrics.accuracy_score(labels_test, pred_test, sample_weight=weights_test))
    else:
        print('  *', cls_type, 'accuracy:', 
              sklearn.metrics.accuracy_score(labels_test, pred_test))

    pred_X = model.predict(X)
    #print(classification_report(Y, pred_X))
    if verbose:
        print(classification_report(labels_test, pred_test))
    return model

###########################################################################################

# Build the linear classifier of a SHAP explainer
def get_SHAP_classifier(label_x0, phi, phi0, x0, EX):
    if len(phi) == 1:
         coef = np.divide(phi[0][label_x0], (x0 - EX), where=(x0 - EX)!=0)
    else:
        coef = np.divide(phi[label_x0], (x0 - EX), where=(x0 - EX)!=0)
    g = sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False)
    g.coef_ = coef
    if not isinstance(phi0, (list, tuple, np.ndarray)):
        g.intercept_ = phi0
    else:
        g.intercept_ = phi0[label_x0]
    return g

###########################################################################################

def eval_whitebox_classifier(R, g, EX, StdX, NormV, x0, label_x0, bb_classifier, wb_name,
                             precision_recalls=False):
    # scale x0 in the ridge model space
    sx0 = np.divide((x0 - EX), StdX, where=np.logical_not(np.isclose(StdX, 0)))
    # sx0 = np.divide((x0 - EX), StdX, where=StdX!=0)
    # compute the p-score of sx0
    sx0_w = np.dot(sx0, g.coef_)
    p_score = sx0_w + g.intercept_

    if linalg.norm(g.coef_) < 1.0e-5 or (abs(sx0_w) < 1.0e-5):# or math.isclose(p_score, 0) or math.isclose(p_score, 1):
        N_sx0_w = np.zeros(len(x0))
        R.wb_plane_dist_x0 = 0.0
    else:
        N_sx0_w = np.multiply(sx0, (0.5 - p_score) / sx0_w)
        R.wb_plane_dist_x0 = p_score / linalg.norm(g.coef_)

    # get the boundary point x1
    sx1 = sx0 + N_sx0_w
    x1 = (sx1 * StdX) + EX

    prob_x1 = bb_classifier([x1])[0] 
    R.wb_class_x1 = 1 if prob_x1[1] > prob_x1[0] else 0
    R.wb_prob_x1_F = prob_x1[0]
    R.wb_prob_x1_T = prob_x1[1]
    R.wb_prob_x1_c0 = prob_x1[label_x0]

    R.wb_local_discr = g.predict([sx0])[0] - R.prob_x0
    R.wb_boundary_discr = g.predict([sx1])[0] - prob_x1[0]

    # build the (scaled) neighborhood of x0
    SNX0 = np.tile(sx0, (NormV.shape[0], 1)) # repeat T times the scaled x1 row
    SNX0 = SNX0 + NormV
    NX0 = (SNX0 * StdX) + EX

    # build the (scaled) neighborhood of x1
    SNX1 = np.tile(sx1, (NormV.shape[0], 1)) # repeat T times the scaled x1 row
    SNX1 = SNX1 + NormV
    NX1 = (SNX1 * StdX) + EX

    # predict the instance classes using the Black-Box and the White-Box classifiers 
    BBY0, WBY0 = bb_classifier(NX0)[:,0], g.predict(SNX0)
    BBY1, WBY1 = bb_classifier(NX1)[:,0], g.predict(SNX1)
    if label_x0 == 1:
        WBY0, WBY1 = 1 - WBY0, 1 - WBY1
    BBCLS0, WBCLS0 = BBY0 > 0.5, WBY0 > 0.5
    BBCLS1, WBCLS1 = BBY1 > 0.5, WBY1 > 0.5

    R.wb_x1_change_score = np.mean(BBCLS1 != label_x0)
    R.wb_avg_bb_nx0 = np.mean(BBY0)
    R.wb_avg_bb_nx1 = np.mean(BBY1)
    R.wb_ratio_x0 = np.mean(BBCLS0)
    R.wb_ratio_x1 = np.mean(BBCLS1)
    R.wb_ratio_wb_x0 = np.mean(WBCLS0)
    R.wb_ratio_wb_x1 = np.mean(WBCLS1)

    try:
        R.wb_fidelity = accuracy_score(BBCLS0, WBCLS0)
        R.wb_prescriptivity = accuracy_score(BBCLS1, WBCLS1)
        R.wb_bal_fidelity = balanced_accuracy_score(BBCLS0, WBCLS0)
        R.wb_bal_prescriptivity = balanced_accuracy_score(BBCLS1, WBCLS1)

        R.wb_fidelity_f1 = f1_score(BBCLS0, WBCLS0)
        R.wb_prescriptivity_f1 = f1_score(BBCLS1, WBCLS1)

        # print(sklearn.metrics.confusion_matrix(BBCLS1, WBCLS1), wb_name)

        if precision_recalls:
            R.wb_precision_x1 = precision_score(BBCLS1, WBCLS1, zero_division=1)
            R.wb_recall_x1 = recall_score(BBCLS1, WBCLS1, zero_division=1)

        # R.wb_fidelity_R2 = g.score(SNX0, BBY0)
        # R.wb_prescriptivity_R2 = g.score(SNX1, BBY1)
    except:
        R.wb_bal_fidelity, R.wb_bal_prescriptivity = 0, 0
        R.wb_fidelity, R.wb_prescriptivity = 0, 0
        # R.wb_fidelity_R2, R.wb_prescriptivity_R2 = 0, 0
        R.wb_fidelity_f1, R.wb_prescriptivity_f1 = 0, 0

    # rename R keys (wb_* -> wb_name_*)
    for key in copy.copy(list(R.__dict__.keys())):
        if key.startswith("wb_"):
            R.__dict__[wb_name + key[2:]] = R.__dict__.pop(key)

    return (x1, sx1)

###########################################################################################

def hinge_loss(x):
    return max(0, 1 - x)

###########################################################################################

class LEAF:
    def __init__(self, bb_classifier, X, Y, class_names, explanation_samples=5000, explainer = 'KernelExplainer'):
        self.bb_classifier = bb_classifier
        self.EX, self.StdX = np.mean(X), np.array(np.std(X, axis=0, ddof=0))
        self.X = X
        self.Y = Y
        self.class_names = class_names
        self.F = X.shape[1] # number of features
        self.explanation_samples = explanation_samples
        self.explainer = explainer

        # SHAP Kernel
        if self.explainer == 'KernelExplainer':
            self.SHAPEXPL = shap.KernelExplainer(self.bb_classifier.predict_proba, shap.kmeans(self.X, 10))
        elif self.explainer == 'TreeExplainer':
            self.SHAPEXPL = shap.TreeExplainer(self.bb_classifier)
        
        elif self.explainer == 'LinearExplainer':
            self.SHAPEXPL = shap.LinearExplainer(self.bb_classifier, shap.maskers.Independent(self.X, len(self.X)))
        else:
            raise Exception("Only KernelExplainer, TreeExplainer, and LinearExplainer are supported") 
        


        self.metrics = None
        self.shap_avg_jaccard_bin = self.shap_std_jaccard_bin = None


    def explain_instance(self, instance, num_reps=50, num_features=4, 
                         neighborhood_samples=10000, use_cov_matrix=False, 
                         verbose=False, figure_dir=None, plot = False):
        npEX = np.array(self.EX)
        cls_proba = self.bb_classifier.predict_proba

        x0 = copy.deepcopy(instance) # instance to be explained
        mockobj = mock.Mock()

        # Neighborhood random samples
        cov_matrix = np.cov(((instance - npEX) / self.StdX).T) if use_cov_matrix else 1.0
        NormV = scipy.stats.multivariate_normal.rvs(mean=np.zeros(self.F), cov=cov_matrix, 
                                                    size=neighborhood_samples, random_state=10)

        # Get the output of the black-box classifier on x0
        output = cls_proba([x0])[0]
        label_x0 = 1 if output[1] >= output[0] else 0
        prob_x0 = output[label_x0]
        prob_x0_F, prob_x0_T = output[0], output[1]
        if verbose:
            print('prob_x0',prob_x0,'   label_x0',self.class_names[label_x0])

        # Prepare instance 
        shap_x0 = (x0 - npEX)

        rows = None
#         progbar = IntProgress(min=0, max=num_reps)
        label = Label(value="")
#         display(HBox([Label("K=%d "%(num_features)), progbar, label]))

        # Explain the same instance x0 multiple times
        for rnum in range(num_reps):
            label.value = "%d/%d" % (rnum+1, num_reps)
            R = mock.Mock() # store all the computed metrics
            R.rnum, R.prob_x0 = rnum, prob_x0
          

            # Explain x0 using SHAP    
            if self.explainer == 'KernelExplainer':
                 shap_phi = self.SHAPEXPL.shap_values(x0, l1_reg="num_features(10)")
                 
                    
            elif self.explainer == "TreeExplainer":
                shap_phi = self.SHAPEXPL.shap_values(x0)
            
            elif self.explainer == "LinearExplainer":
                shap_phi = [np.array(self.SHAPEXPL.shap_values(x0))]
                
                
          
            shap_phi0 = self.SHAPEXPL.expected_value
            

            # Take only the top @num_features from shap_phi
            argtop = np.argsort(np.abs(shap_phi[0]))
            for k in range(len(shap_phi)):
                shap_phi[k][ argtop[:(self.F-num_features)] ] = 0

            # Recover the SHAP classifiers
            R.shap_g = get_SHAP_classifier(label_x0, shap_phi, shap_phi0, x0, self.EX)

            #----------------------------------------------------------
            # Evaluate the white box classifiers
         
            ES = eval_whitebox_classifier(R, R.shap_g, npEX, np.ones(len(x0)), 
                                          NormV * self.StdX, x0, label_x0, cls_proba, "shap", 
                                          precision_recalls=True)

            R.shap_local_discr = np.abs(R.shap_g.predict([shap_x0])[0] - prob_x0)
        
            # Indices of the most important features, ordered by their absolute value
            R.shap_argtop = np.argsort(np.abs(R.shap_g.coef_))

            # get the K most common features in the explanation of x0
            R.mcf_shap = tuple([R.shap_argtop[-k] for k in range(num_features)])

            # Binary masks of the argtops
            R.shap_bin_expl = np.zeros(self.F)
            R.shap_bin_expl[np.array(R.mcf_shap)] = 1

            # get the appropriate R keys
            R_keys = copy.copy(R.__dict__)
            for key in copy.copy(list(R_keys.keys())):
                if key.startswith("wb_"):
                    R_keys[wb_name + key[2:]] = R_keys.pop(key)
                elif key in mockobj.__dict__:
                    del R_keys[key]

            rows = pd.DataFrame(columns=R_keys) if rows is None else rows
            rows = rows.append({k:R.__dict__[k] for k in R_keys}, ignore_index=True)
#             progbar.value += 1

#         label.value += " Done."

        # use the multiple explanations to compute the LEAF metrics
        # display(rows)

        # Jaccard distances between the various explanations (stability)
        shap_jaccard_mat = 1 - pdist(np.stack(rows.shap_bin_expl, axis=0), 'jaccard')
        self.shap_avg_jaccard_bin, self.shap_std_jaccard_bin = np.mean(shap_jaccard_mat), np.std(shap_jaccard_mat)



        # store the metrics for later use
        self.metrics = rows
        if plot:
            def leaf_plot(stability, method):
                fig, ax1 = plt.subplots(figsize=(6, 2.2))
                data = [ stability.flatten(),
                         1 - rows[method + '_local_discr'], 
                         rows[method + '_fidelity_f1'], 
                         # rows[method + '_prescriptivity_f1'],
                         # rows[method + '_bal_prescriptivity' ],
                         1 - 2 * np.abs(rows[method + '_boundary_discr' ]) ]


                # color = 'tab:red'
                ax1.tick_params(axis='both', which='major', labelsize=12)
                ax1.set_xlabel('distribution')
                ax1.set_ylabel('LEAF metrics', color='black', fontsize=15)
                ax1.boxplot(data, vert=False, widths=0.7)
                ax1.tick_params(axis='y', labelcolor='#500000')
                ax1.set_yticks(np.arange(1, len(data)+1))
                ax1.set_yticklabels(['Stability', 'Local Concordance', 'Fidelity', 'Prescriptivity'])
                ax1.set_xlim([-0.05,1.05])
                ax1.invert_yaxis()

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.tick_params(axis='both', which='major', labelsize=12)
                ax2.set_ylabel('Values', color='#000080')  # we already handled the x-label with ax1
                ax2.boxplot(data, vert=False, widths=0.7)
                # ax2.boxplot([np.mean(d) for d in data], color=color)
                ax2.tick_params(axis='y', labelcolor='#000080')
                ax2.set_yticks(np.arange(1, len(data)+1))
                ax2.set_yticklabels([ "  %.3f ± %.3f  " % (np.mean(d), np.std(d)) for d in data])
                ax2.invert_yaxis()

                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                if figure_dir is not None:
                    imgname = figure_dir+method+"_leaf.pdf"
                    print('Saving', imgname)
                    plt.savefig(imgname, dpi=150, bbox_inches='tight')
                plt.show()

            # Show SHAP explanation
            display(HTML("<h2>SHAP</h2>"))
            display(shap.force_plot(shap_phi0[label_x0], shap_phi[label_x0], x0))
            leaf_plot(shap_jaccard_mat, 'shap')

        prescription = False
        if prescription:
            print("====================================================")
            shap_x1, shap_sx1 = ES

            print('SHAP accuracy %f balanced_accuracy %f precision %f recall %f' % 
                  (rows.shap_prescriptivity.mean(), rows.shap_bal_prescriptivity.mean(),
                   rows.shap_precision_x1.mean(), rows.shap_recall_x1.mean()))


            shap_diff = (rows.iloc[-1].shap_g.coef_ != 0) * (shap_x1 - x0)

            print('shap_diff\n', shap_diff)


            shap_output_x1 = cls_proba([shap_x1])[0]
            shap_label_x1 = 1 if shap_output_x1[1] >= shap_output_x1[0] else 0

            print("SHAP(x1) prob =", shap_output_x1)

            # df = pd.DataFrame([x0, x0 + shap_diff], index=['x', 'x\'']).round(2)
            # display(df.T.iloc[:math.ceil(F/2),:])
            # display(df.T.iloc[math.ceil(F/2):,:])


            # Show SHAP explanation
            shap_phi = SHAPEXPL.shap_values(shap_x1, l1_reg="num_features(10)")
            shap_phi0 = SHAPEXPL.expected_value
            argtop = np.argsort(np.abs(shap_phi[0]))
            for k in range(len(shap_phi)):
                shap_phi[k][ argtop[:(F-num_features)] ] = 0
            display(shap.force_plot(shap_phi0[shap_label_x1], shap_phi[shap_label_x1], shap_x1))

    def get_R(self):
        return self.metrics


    #------------------------------------------#

    def get_shap_stability(self):
        assert self.metrics is not None
        return self.shap_avg_jaccard_bin

    def get_shap_local_concordance(self):
        assert self.metrics is not None
        return hinge_loss(np.mean(self.metrics.shap_local_discr))

    def get_shap_fidelity(self):
        assert self.metrics is not None
        return np.mean(self.metrics.shap_fidelity_f1)

    def get_shap_prescriptivity(self):
        assert self.metrics is not None
        return hinge_loss(np.mean(2 * np.abs(self.metrics.shap_boundary_discr)))

    #------------------------------------------#
    #Iterate through the whole dataset, making explanations of each instance/sample
    def explain_dataset(self, num_reps = 50, num_features = None, n_samples = None):
        if num_features == None:
            num_features = np.int64(np.round(len(self.X.columns)*0.1))
        if n_samples == None:
            n_samples = len(self.X)
        
        #SHAP lists for iteration
        SHAP_stability = []
        SHAP_local_concordance = []
        SHAP_fidelity = []
        SHAP_prescriptivity = []

        #Implement fastprogress' master_bar to keep track of the for_loop
        mb = progress_bar(range(n_samples))
        for i in mb:
            eval_IND = self.X.iloc[i,:]
            self.explain_instance(eval_IND, num_reps = num_reps, num_features = num_features)



            SHAP_stability.append(self.get_shap_stability())
            SHAP_local_concordance.append(self.get_shap_local_concordance())
            SHAP_fidelity.append(self.get_shap_fidelity())
            SHAP_prescriptivity.append(self.get_shap_prescriptivity())


        return pd.DataFrame({'Stability':SHAP_stability, 'Local Concordance':SHAP_local_concordance, 'Fidelity':SHAP_fidelity, 'Prescriptivity':SHAP_prescriptivity})

