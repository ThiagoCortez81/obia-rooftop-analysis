from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from osgeo import gdal, ogr
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# ROC Line
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


# naip_fn = '../dados/Dados_GEO/ortofoto.tif'
naip_fn = '../dados/Dados_Finais/TFG_THIAGO/bairro.tif'
segments_fn = './result/amostra1/classified_fold_1.tif'

driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)

# test_fn = './result/truth_data.shp'
test_fn = './result/test_data.shp'
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()

driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize,
                          naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray()

pred_ds = gdal.Open(segments_fn)
pred = pred_ds.GetRasterBand(1).ReadAsArray()
pred[pred > 2] = 2


# Excluding all zero values from truth data (classified.tif)
idx = np.nonzero(truth)

# pixel by pixel matrix and not segment by segment
cm = confusion_matrix(truth[idx], pred[idx], labels=[
                      1, 2])  # Binary (telhado, ñ telhado)
tn, fp, fn, tp = cm.ravel()
negatives = sum([tn, fn])
positives = sum([fp, tp])

true_positive_rate = tp/positives

false_positive_rate = fp/negatives

true_negative_rate = tn/negatives

false_negative_rate = fn/positives

error_rate = (fp+fn)/(positives+negatives)

accuracy1 = (tp+tn)/(negatives + positives)

precision = tp/(tp+fp)

sensitivity1 = tp/(tp+fn)

specificity = tn/(tn+fp)

f1 = (2*tp) / ((2*tp) + fp + fn)

# Reference https://towardsdatascience.com/understanding-the-confusion-matrix-and-how-to-implement-it-in-python-319202e0fe4d
clf = SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(
    clf, truth[idx].reshape(-1, 1), pred[idx].reshape(-1, 1), cv=5)


print('\n\n\n\n============================ RESULTADOS ============================')
print('True Positive Rate', truncate(true_positive_rate * 100, decimals=2), '%')
print('False Negative Rate', truncate(
    false_negative_rate * 100, decimals=2), '%')
print('True Negative Rate', truncate(true_negative_rate * 100, decimals=2), '%')
print('False Positive Rate', truncate(
    false_positive_rate * 100, decimals=2), '%')
print('Error Rate:', truncate(error_rate * 100, decimals=2), '%')
print('Accuracy : ', truncate(accuracy1 * 100, decimals=2), '%')
print('Precision : ', precision)
print('Sensitivity/Recall : ', sensitivity1)
print('Specificity : ', specificity)
print('F1 score : ', f1)
print("{}% de acurácia com desvio padrão de {}".format(
    truncate(scores.mean() * 100, decimals=2), scores.std()))


X_train, X_test, y_train, y_test = train_test_split(truth[idx].reshape(-1, 1), pred[idx].reshape(-1, 1), test_size=0.3)

clf_tree = DecisionTreeClassifier()

clf_tree.fit(X_train, y_train) 
y_score1 = clf_tree.predict_proba(X_test)[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1, pos_label=2)

auc = roc_auc_score(y_test, y_score1)
print('AUC: %.3f' % auc)
print('====================================================================')

plt.subplots(1, figsize=(5,5))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
