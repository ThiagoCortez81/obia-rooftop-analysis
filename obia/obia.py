import os
import numpy as np
import math
import time
import scipy
import csv
from osgeo import gdal, ogr
from skimage import exposure
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# haralick
import mahotas as mt
import geopandas as gpd

# Kfold - Cross Validation
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.svm import SVC

# ROC Line
from sklearn.tree import DecisionTreeClassifier

get_from_cache = True

properties_csv_fn = './result/properties.csv'
# naip_fn = '../dados/Dados_GEO/ortofoto.tif'
naip_fn = '../dados/Dados_Finais/TFG_THIAGO/bairro.tif'
segments_fn = './result/segments.tif'

driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)
nbands = naip_ds.RasterCount
band_data = []

print('bands', nbands, 'rows', naip_ds.RasterYSize,
      'columns', naip_ds.RasterXSize)

print('naip_ds.GetGeoTransform()', naip_ds.GetGeoTransform())

# Conferir BANDAS abaixo
for i in range(1, nbands+1):
    band = naip_ds.GetRasterBand(i).ReadAsArray()

    band_data.append(band)

band_data = np.dstack(band_data)
img = exposure.rescale_intensity(band_data)  # Verificar se é necessário

# another approach to segment image


def segment_image(img2):
    # move imports to top of the image
    from scipy import ndimage as ndi
    from skimage.morphology import disk
    from skimage.filters import rank
    from skimage.util import img_as_ubyte

    image = img_as_ubyte(img2[:, :, 0])

    # denoise image
    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    return watershed(gradient, markers)


# do segmentation
# seg_start = time.time()
# segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=500)
# segments = felzenszwalb(img, scale=500)
# segments = slic(img, n_segments=2108, compactness=0.001)
# segments = quickshift(img, convert2lab=False)
# segments = segment_image(img)

# save the segments to raster
# segments_ds = driverTiff.Create(
#     segments_fn, naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
# segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
# segments_ds.SetProjection(naip_ds.GetProjectionRef())
# segments_ds.GetRasterBand(1).WriteArray(segments)
# segments_ds = None

# raise Exception('fds')

# print('segments complete', time.time() - seg_start)

driverTiff = gdal.GetDriverByName('GTiff')
segment_ds = gdal.Open(segments_fn)

segments = segment_ds.GetRasterBand(1).ReadAsArray()

properties_csv_ds = open(properties_csv_fn, 'r', encoding='UTF8', newline='')
if not get_from_cache:
    properties_csv_ds = open(properties_csv_fn, 'w',
                             encoding='UTF8', newline='')
    csv_writer = csv.writer(properties_csv_ds)
    # writing header
    csv_writer.writerow(['nobs', 'min', 'max', 'mean', 'variance',
                        'skewness', 'kurtosis', 'asm', 'var', 'idm', 'ent'])


def extract_haralick(segment_pixels_for_band):
    # return []
    # calculate haralick texture features for 4 types of adjacency
    # for cmat in segment_pixels_for_band:
    #   if not cmat.sum():
    #     print('Banda vazia, ignorando haralick')
    #     return [0,0,0,0]

    try:
        arr_textures = mt.features.haralick(
            segment_pixels_for_band, distance=1, return_mean=True, return_mean_ptp=False)  # test return_mean_ptp (min - max) // change distance to 3

        # take the mean of it and return it
        # arr_textures = textures.mean(axis=0)
        # values = asm, var, idm, ent
        return [arr_textures[0], arr_textures[3], arr_textures[4], arr_textures[8]]
    except:
        # case haralick return ValueError: mahotas.haralick_features: the input is empty. Cannot compute features!
        return [0, 0, 0, 0]


def cached_segment_features(custom_items={'asm': True, 'var': True, 'idm': True, 'ent': True}):
    csv_reader = csv.reader(properties_csv_ds)
    header = next(csv_reader)
    print('getting the following data:', header)
    print('custom_items', custom_items)
    print('custom_items', custom_items.get('asm'))

    data = []
    temp_data = []
    data_counter = 0
    for row in csv_reader:
        parsed_row = [int(row[0]), int(row[1]), int(row[2]), float(
            row[3]), float(row[4]), float(row[5]), float(row[6])]

        # , float(row[7]), float(row[8]), float(row[9]), float(row[10])

        if custom_items.get('asm'):
            parsed_row.append(float(row[7]))
        if custom_items.get('var'):
            parsed_row.append(float(row[8]))
        if custom_items.get('idm'):
            parsed_row.append(float(row[9]))
        if custom_items.get('ent'):
            parsed_row.append(float(row[10]))

        temp_data.append(parsed_row)
        data_counter += 1
        if data_counter >= 3:
            data.append(parsed_row)
            temp_data = []
            data_counter = 0

    properties_csv_ds.close()
    return data


def segment_features(segment_pixels, id, get_from_cache=True):
    features = []
    npixels, nbands = segment_pixels.shape

    features_haralick = extract_haralick(segment_pixels)

    for b in range(nbands):

        # Composto por: (nobs, minmax=(0, 0), mean, variance, skewness, kurtosis, asm, var, idm, ent)
        stats = scipy.stats.describe(segment_pixels[:, b])
        splitted_minmax = stats.minmax  # index 0 will be the min and 1 the max
        stats_to_write = [stats.nobs, splitted_minmax[0], splitted_minmax[1],
                          stats.mean, stats.variance, stats.skewness, stats.kurtosis, features_haralick[0], features_haralick[1], features_haralick[2], features_haralick[3]]

        # add haralick aqui
        csv_writer.writerow(stats_to_write)
        print('features_haralick', features_haralick)
        # includes min,max,mean, variance, skewness e kurtosis
        band_stats = list(stats.minmax) + list(stats)[2:]
        # adding haralick metrics (ALTERAR para considerar alguns dados apenas)
        # band_stats.extend(features_haralick)
        print('band_stats for id:{} is: {}'.format(id, band_stats))
        if npixels == 1:
            # in this case the variance = nan, change it to 0.0
            band_stats[3] = 0.0
        features += band_stats

        # print(u'band_stats', band_stats)

    # exportar features csv (numero feature, vetor de atributos [stats])

    return features


obj_start = time.time()
segment_ids = np.unique(segments)
print('segment_ids', segment_ids)
objects = []
object_ids = []

for id in segment_ids:
    if not get_from_cache:
        segment_pixels = img[segments == id]
        object_features = segment_features(segment_pixels, id)
        objects.append(object_features)
    object_ids.append(id)

if get_from_cache:
    custom_items = {'asm': True, 'var': True, 'idm': True, 'ent': True}
    objects = cached_segment_features(custom_items)

print('created', len(objects), 'objects with', len(
    objects[0]), 'variables in', time.time()-obj_start, 'seconds')

# Split testing data
# Formatting classes to match 1,2 format
kf = KFold(n_splits=5)
dataset_fn = '../dados/Dados_Finais/TFG_THIAGO/Shapes/amostras3.shp'
gdf = gpd.read_file(dataset_fn)

class_names = gdf['classe'].unique()
print('class_names', class_names)
class_ids = np.arange(class_names.size) + 1
print('class_ids', class_ids)

gdf['id'] = gdf['classe'].map(dict(zip(class_names, class_ids)))

class_names = gdf['classe'].unique()
print('class_names', class_names)
class_ids = np.arange(class_names.size) + 1
print('class_ids', class_ids)

fold_n = 1
fold_data = []
for train_index, test_index in kf.split(gdf):
    gdf_train_fn = './temp/amostra3/train_data.shp'
    gdf_test_fn = './temp/amostra3/test_data.shp'
    if not os.path.isfile('./result/amostra3/OBIA_RUGOSIDADE_1/classified_fold_{}.tif'.format(fold_n)):
        obj_start = time.time()
        # train_fn = './result/train_data.shp's
        # train_ds = ogr.Open(train_fn)
        # lyr = train_ds.GetLayer()

        gdf_train = gdf.iloc[train_index]
        gdf_train.to_file(gdf_train_fn)

        gdf_test = gdf.iloc[test_index]
        gdf_test.to_file(gdf_test_fn)

        train_ds = ogr.Open(gdf_train_fn)
        lyr = train_ds.GetLayer()

        driver = gdal.GetDriverByName('MEM')
        target_ds = driver.Create('', naip_ds.RasterXSize,
                                  naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
        target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
        target_ds.SetProjection(naip_ds.GetProjection())
        options = ['ATTRIBUTE=id']
        gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

        ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

        classes = np.unique(ground_truth)[1:]
        print('class values', classes)

        segments_per_class = {}

        for klass in classes:
            segments_of_class = segments[ground_truth == klass]
            segments_per_class[klass] = set(segments_of_class)
            print('Training segments for class',
                  klass, ':', len(segments_of_class))

        intersection = set()
        accum = set()

        # for class_segments in segments_per_class.values():
        #     intersection |= accum.intersection(class_segments)
        #     accum |= class_segments

        # assert len(intersection) == 0, 'Segments represent multiple classes'

        train_img = np.copy(segments)
        threshold = train_img.max() + 1

        for klass in classes:
            class_label = threshold + klass
            for segment_id in segments_per_class[klass]:
                train_img[train_img == segment_id] = class_label

        train_img[train_img <= threshold] = 0
        train_img[train_img > threshold] -= threshold

        training_objects = []
        training_labels = []

        print('enumerate', enumerate(objects))

        for klass in classes:
            class_train_object = [v for i, v in enumerate(
                objects) if segment_ids[i] in segments_per_class[klass]]
            training_labels += [klass] * len(class_train_object)
            training_objects += class_train_object
            print('Training objects for class',
                  klass, ':', len(class_train_object))

        print('training_objects', training_objects)

        classifier = RandomForestClassifier(n_jobs=-1)
        # classifier = MLPClassifier(
        #     solver='lbfgs', alpha=1e-5, max_iter=99999999, random_state=1)
        # classifier = SVC(random_state=42)
        print('Fitting random forest classifier')
        classifier.fit(training_objects, training_labels)
        print('Predicting classifier')
        predicted = classifier.predict(objects)

        clf = np.copy(segments)

        for segment_id, klass in zip(segment_ids, predicted):
          clf[clf == segment_id] = klass


        print('Klassifying done! Lasted ', time.time()-obj_start, 'secs')

        mask = np.sum(img, axis=2)
        mask[mask > 0.0] = 1.0
        mask[mask == 0.0] = -1.0
        clf = np.multiply(clf, mask)
        clf[clf < 0] = -9999.0

        clfds = driverTiff.Create('./result/amostra3/OBIA_RUGOSIDADE_1/classified_fold_{}.tif'.format(fold_n),
                                  naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
        # clfds = driverTiff.Create('./result/classified_normal.tif', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
        # clfds = driverTiff.Create('./result/classified_haralick.tif', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
        # clfds = driverTiff.Create('./result/classified_haralick_mean.tif',
        #                           naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
        # clfds = driverTiff.Create('./result/classified_haralick_slic.tif',
        #                           naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
        # clfds = driverTiff.Create('./result/classified_haralick_watershed.tif',
        #                           naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32)
        clfds.SetGeoTransform(naip_ds.GetGeoTransform())
        clfds.SetProjection(naip_ds.GetProjection())
        clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
        clfds.GetRasterBand(1).WriteArray(clf)
        clfds = None

    def confusion_matrix_for_classification():
        def truncate(n, decimals=0):
            multiplier = 10 ** decimals
            return int(n * multiplier) / multiplier

        naip_ds = gdal.Open(naip_fn)

        # test_fn = './result/truth_data.shp'
        test_ds = ogr.Open(gdf_test_fn)
        lyr = test_ds.GetLayer()

        driver = gdal.GetDriverByName('MEM')
        target_ds = driver.Create('', naip_ds.RasterXSize,
                                  naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
        target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
        target_ds.SetProjection(naip_ds.GetProjection())
        options = ['ATTRIBUTE=id']
        gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

        truth = target_ds.GetRasterBand(1).ReadAsArray()

        pred_ds = gdal.Open(
            './result/amostra3/OBIA_RUGOSIDADE_1/classified_fold_{}.tif'.format(fold_n))
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

        accuracy2 = (1/2) * (tp/positives + tn/negatives)

        # mcc = ((tp*tn) - (fp*fn)) / \
        #     (math.sqrt(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))

        precision = tp/(tp+fp)

        sensitivity1 = tp/(tp+fn)

        specificity = tn/(tn+fp)

        f1 = (2*tp) / ((2*tp) + fp + fn)

        # Referências https://towardsdatascience.com/understanding-the-confusion-matrix-and-how-to-implement-it-in-python-319202e0fe4d

        # ADD mais métricas para a matriz e confusão

        # Accuracy :  0.8901515151515151
        # Sensitivity :  0.8660714285714286
        # Specificity :  0.9078947368421053

        # Cross validation
        clf = SVC(kernel='linear', C=1, random_state=42)
        scores = cross_val_score(
            clf, truth[idx].reshape(-1, 1), pred[idx].reshape(-1, 1), cv=5)

        print(
            '\n\n\n\n============================ RESULTADOS ============================')
        print('True Positive Rate', truncate(
            true_positive_rate * 100, decimals=2), '%')
        print('False Negative Rate', truncate(
            false_negative_rate * 100, decimals=2), '%')
        print('True Negative Rate', truncate(
            true_negative_rate * 100, decimals=2), '%')
        print('False Positive Rate', truncate(
            false_positive_rate * 100, decimals=2), '%')
        print('Error Rate:', truncate(error_rate * 100, decimals=2), '%')
        print('Accuracy : ', truncate(accuracy1 * 100, decimals=2), '%')
        print('Balanced Accuracy (MCC) : ', truncate(
            accuracy2 * 100, decimals=2), '%')
        print('Precision : ', precision)
        print('Sensitivity/Recall : ', sensitivity1)
        print('Specificity : ', specificity)
        print('F1 score : ', f1)
        print("{}% de acurácia com desvio padrão de {}".format(
            truncate(scores.mean() * 100, decimals=2), scores.std()))

        X_train, X_test, y_train, y_test = train_test_split(
            truth[idx].reshape(-1, 1), pred[idx].reshape(-1, 1), test_size=0.3)

        clf_tree = DecisionTreeClassifier()

        clf_tree.fit(X_train, y_train)
        y_score1 = clf_tree.predict_proba(X_test)[:, 1]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(
            y_test, y_score1, pos_label=2)

        auc = roc_auc_score(y_test, y_score1)
        print('AUC: %.3f' % auc)
        print('====================================================================')
        # fold_data.append([fold_n, true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate,
        #                   error_rate, accuracy1, accuracy2, precision, sensitivity1, specificity, f1, auc])
        fold_data.append([accuracy1, accuracy2, precision, sensitivity1, specificity, auc, f1])

    confusion_matrix_for_classification()
    fold_n += 1

print('All gathered data:', fold_data)
with open('./result/amostra3/OBIA_RUGOSIDADE_1/data_matrix_random_forest.txt', 'w+') as testfile:
    for row in fold_data:
        testfile.write(' '.join([str(a) for a in row]) + '\n')
print('YAY! It\'s done.')
