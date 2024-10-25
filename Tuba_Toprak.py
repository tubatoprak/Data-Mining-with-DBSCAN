import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from sklearn import metrics



def PCA_Spotify(d , numb):
    normalized = d - np.mean(d , axis = 0)     
    covmat = np.cov(normalized , rowvar = False)     
    eigen_values , eigen_vectors = np.linalg.eigh(covmat)     
    sorindex = np.argsort(eigen_values)[::-1]
    evalues = eigen_values[sorindex]
    evectors = eigen_vectors[:,sorindex]     
    eigenvector_subset = evectors[:,0:numb]     
    pcad = np.dot(eigenvector_subset.transpose(),normalized.transpose()).transpose()
     
    return pcad



# DATA READING
df = pd.read_csv("music_genre.csv",sep = ",")

# DATA CLEANING
print(df.isna().sum()) 
df = df.drop(columns=['Unnamed: 0'],axis =1) 

# genre tiplerini sayısal verilere dönüştürdük
label_encoder = LabelEncoder()
keep_genres = df['Genre']
df['Genre']= label_encoder.fit_transform(df['Genre'])

# genre attibute unun diğer özelliklerle olan ilişkisine(korelasyonuna bakılır) (2)
plt.figure(figsize = (6, 4), dpi = 100)
df.corr()['Genre'].sort_values().plot(kind  = 'bar')

# CORRELATION MATRIX
x, y = plt.subplots(figsize = (10,10))
sns.heatmap(df.corr(), annot=True, fmt='.1f',
            ax=y, cmap='Spectral', vmin=-1, vmax=1)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Matrix', size=10);
plt.show()

# FEATURE SELECTION  -> sadece korelasyon değeri yüksek olan özelliklerin subset ini oluşturuyoruz.

df = df.drop(columns=["instrumentalness","acousticness","mode","valence","danceability",
                      "duration_ms","speechiness"],axis =1) 


# feature selection işleminden sonra seçilen özelliklerin genre tipi ile olan korelasyonu
plt.figure(figsize = (6, 4), dpi = 100)
df.corr()['Genre'].sort_values().plot(kind  = 'bar')
plt.show()


# tür ve bu türlerin çevrildiği sayısal verilerin dictionarysini oluşturuyoruz
zip_iterator = zip(keep_genres,df["Genre"])
a_dictionary = dict(zip_iterator)
print(a_dictionary)

# müzik tiplerinin(sayısal olarak) dağlımlarının grafiği
sns.countplot(df["Genre"])
plt.title('Distribution of Genres', size=12);
plt.show()




withoutGenres = df.drop(["Genre","artist_names","song_title (censored)","artists_id","release_date"],axis = 1)
withGenres = df.Genre
# PCA(Principal Component Analysis) yöntemi ile FEATURE EXTRACTION 

withoutGenres = StandardScaler().fit_transform(withoutGenres)
PCA_DATAFRAME = PCA_Spotify(withoutGenres,2)  # düşüreceğimiz boyut 2
# PCA1 boyutu düşürülmüş datanın ilk sütunu
dfPCA = pd.DataFrame(data = PCA_DATAFRAME,columns = ['PCA1', 'PCA2'])



#BOXPLOT ile outlier tespiti
plt1 = sns.boxplot(dfPCA["PCA1"])
plt.show()
plt2 = sns.boxplot(dfPCA['PCA2'])
plt.show()



X_train,X_test,y_train,y_test=train_test_split(withoutGenres,
                                               withGenres,test_size=0.1,
                                               random_state=42) 

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# classification
print("Classification accuracy for RandomForestClassifier\n")

model=RandomForestClassifier(n_estimators=60, random_state=0)
model.fit(X_train,y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print('RandomForestClassifier', end = ":")
print(" %",acc*100)


# CLUSTERING after PCA
#DBSCAN
#df = StandardScaler().fit_transform(dfPCA)
dbs = DBSCAN(eps=0.3, min_samples=30).fit_predict(dfPCA)
plt.title("DBSCAN Clustering")
plt.scatter(dfPCA["PCA1"],dfPCA["PCA2"], c=dbs)
sc = metrics.silhouette_score(dfPCA, dbs, metric='euclidean')
print('Silhouette Coefficient of DBSCAN Clustering: %.4f\n' % sc)
 







