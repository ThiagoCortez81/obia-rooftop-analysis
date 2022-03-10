import numpy as np
import geopandas as gpd
import pandas as pd

gdf = gpd.read_file('../dados/Dados_Finais/TFG_THIAGO/Shapes/amostras1.shp')
class_names = gdf['classe'].unique()
print('class_names', class_names)
class_ids = np.arange(class_names.size) + 1
print('class_ids', class_ids)

df = pd.DataFrame({'classe': class_names, 'id': class_ids})
df.to_csv('./result/class_lookup.csv')

print('gdf without ids', gdf.head())
gdf['id'] = gdf['classe'].map(dict(zip(class_names, class_ids)))
print('gdf with ids', gdf.head())

gdf_train = gdf.sample(frac=0.7)
gdf_test = gdf.drop(gdf_train.index)

print('gdf shape', gdf.shape, 'training shape', gdf_train.shape, 'test', gdf_test.shape)

gdf_train.to_file('./result/amostras1/train_data.shp')
gdf_test.to_file('./result/amostras1/test_data.shp')
