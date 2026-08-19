## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
from google.colab import drive drive.mount('/content/drive')

ls drive/MyDrive/'Colab Notebooks'/
ENCODEING 
import pandas as pd import numpy as np

df=pd.read_csv('drive/MyDrive/Data Science/Encoding Data.csv') df<img width="356" height="398" alt="395072142-6f208659-9414-4ca6-8d8d-d1ad470e85b2" src="https://github.com/user-attachments/assets/362edf7d-cabf-432a-b9fe-c56e31489889" />

ORDINAL ENCODEING
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pm= ['Hot','Warm','Cold']

en1 = OrdinalEncoder(categories = [pm])

en1.fit_transform(df[["ord_2"]]
<img width="259" height="220" alt="395072227-1de5b2a7-ba89-44aa-938a-25727bee77be" src="https://github.com/user-attachments/assets/cf2805a0-efd1-4bfe-b734-22c3ecdb59e0" />
df['bo2']=en1.fit_transform(df[["ord_2"]]) df

LABLE ENCODEING
<img width="418" height="401" alt="395072433-8e740dc1-6a58-45b2-8746-343acc25891c" src="https://github.com/user-attachments/assets/b6680899-3a94-4638-939e-7f904bbc9452" />

ONE HOT ENCODER
from sklearn.preprocessing import OneHotEncoder

One=OneHotEncoder(sparse_output=False) df2=df.copy()

enc=pd.DataFrame(One.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1) df2
<img width="518" height="410" alt="395072549-1190f9b5-aada-44de-a7cc-57989365fdbb" src="https://github.com/user-attachments/assets/67612e56-8a20-4088-b8b5-2e5c58448aca" />
pd.get_dummies(df2,columns=["nom_0"])

<img width="772" height="413" alt="395072632-bda17d86-7261-4a4b-986c-d57b9059c842" src="https://github.com/user-attachments/assets/c595a9f6-4e4a-4ee9-9d3c-e77b7cfc4a6c" />
BINARY ENCODER
pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("drive/MyDrive/Data Science/data.csv") df
<img width="579" height="411" alt="395072779-6b014e06-34e2-4331-8a46-57973e44f2d2" src="https://github.com/user-attachments/assets/6ef581e9-3daf-4856-9110-ab725e337ab1" />
be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2']) dfb=pd.concat([df,nd],axis=1) dfb1=df.copy()

dfb1
<img width="588" height="396" alt="395072864-31cceed1-5b52-450c-9912-b698178acd42" src="https://github.com/user-attachments/assets/d4ea6f60-5dfd-483a-ab4c-01240ee90cfc" />
TARGET ENCODER
from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"]) cc=pd.concat([cc,new],axis=1) cc
TARGET ENCODER
from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"]) cc=pd.concat([cc,new],axis=1) cc
TARGET ENCODER
from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"]) cc=pd.concat([cc,new],axis=1) cc


<img width="661" height="397" alt="395072920-2e6ed56b-cf26-4745-b794-233714d7b040" src="https://github.com/user-attachments/assets/237b0f58-4c0b-497d-b74b-8965ff8fadfc" />

TRANSFORMATION
import numpy as np import pandas as pd import matplotlib.pyplot as plt import statsmodels.api as sm import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv('drive/MyDrive/Data Science/Data_to_Transform.csv') df
<img width="840" height="494" alt="395073036-2f37aaee-e7d5-4461-b1c6-8a1b68986025" src="https://github.com/user-attachments/assets/03ae851a-8eb6-4794-bdc6-a926bc3f5b2b" />
<img width="392" height="238" alt="395073162-524e38c7-1a45-4492-8e59-688e1722dd1a" src="https://github.com/user-attachments/assets/bcfa63ff-1f43-46d6-8e24-b88cf053013b" />
np.log(df["Highly Positive Skew"])


<img width="362" height="480" alt="395073217-fb23f998-cda4-44d9-8c3b-e60ae6aa150f" src="https://github.com/user-attachments/assets/ce88a573-77d1-47e5-8aa8-62090adb0cb2" />
np.reciprocal(df["Moderate Positive Skew"])
<img width="389" height="479" alt="395073283-583bba9d-772a-4b51-8462-db36c671e173" src="https://github.com/user-attachments/assets/24d2a20b-987d-48ae-91ff-8ec481ca8125" />
np.sqrt(df["Highly Positive Skew"])
<img width="354" height="482" alt="395073350-b1af8c3b-077b-4eec-9e3f-4d1bb94b4bb1" src="https://github.com/user-attachments/assets/1f5637ad-048f-48f1-8306-b629674513e0" />
np.square(df["Highly Positive Skew"])
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"]) df

<img width="847" height="495" alt="395073626-c03255d2-250b-45e3-9f06-dd8bdd5ed42a" src="https://github.com/user-attachments/assets/40cf9662-9ce9-4a26-8605-cb2402df70bb" />
<img width="847" height="495" alt="395073626-c03255d2-250b-45e3-9f06-dd8bdd5ed42a" src="https://github.com/user-attachments/assets/63ac6a5a-5dcf-424e-8dfd-6666070cd652" />
df["Moderate Negative Skew_yeojohnson"], lmbda = stats.yeojohnson(df["Moderate Negative Skew"])

df.skew()
<img width="493" height="319" alt="395073704-28318a7a-2013-4033-99f7-6cc86cc858e0" src="https://github.com/user-attachments/assets/d187e828-63fb-4614-a685-28e7d3808a07" />
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()<img width="506" height="335" alt="395073801-75da5d40-9fe2-4d20-be93-4e6cf7a20999" src="https://github.com/user-attachments/assets/cc54b547-edef-48e4-b3d4-141bd2226a01" />

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) df
<img width="1805" height="486" alt="395073942-93e00d44-6da9-4c98-9de7-64534f15eab5" src="https://github.com/user-attachments/assets/1427c65b-9d05-432a-aa2a-735313fc433a" />

import seaborn as sns import statsmodels.api as sm import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45') plt.show()
<img width="775" height="525" alt="395074081-9e6c45bc-6fc0-484f-801f-ff4fbb25a3c8" src="https://github.com/user-attachments/assets/8f6c69e9-b69b-47f1-bfd4-eb3e269a08ab" />
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') plt.show()

<img width="775" height="511" alt="395074168-1f89e6dd-bb04-41e4-a783-516d8b5cc749" src="https://github.com/user-attachments/assets/247b4623-aedd-4eed-ac95-41a048b6ba0d" />


from sklearn.preprocessing import QuantileTransformer qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45') plt.show()

<img width="743" height="497" alt="395074253-71336d4a-904b-4d33-8b5c-d7757c8bbc46" src="https://github.com/user-attachments/assets/510c99ec-fe06-424e-bd88-bfc89e73743c" />
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]]) sm.qqplot(df['Highly Negative Skew'],line='45') plt.show()


<img width="742" height="495" alt="395074336-2f08652a-ad55-4eab-947d-70e9f0c69f12" src="https://github.com/user-attachments/assets/c5322900-aad7-45a2-a44d-b4c06b30d4be" />

<img width="726" height="514" alt="395074465-37f1fa0f-7df1-4b78-98d3-79b25baf66f2" src="https://github.com/user-attachments/assets/7af9340d-5d19-4922-ab09-7a31e4184a06" />
dt=pd.read_csv("drive/MyDrive/Data Science/titanic_dataset.csv")

from sklearn.preprocessing import QuantileTransformer qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]]) sm.qqplot(dt['Age'],line='45') plt.show()

<img width="724" height="494" alt="395074586-7fc2cc81-c072-4222-ab5c-ac68368f0bbd" src="https://github.com/user-attachments/assets/535a3c26-3525-4271-9314-ce7cfbc5b473" />
<img width="740" height="514" alt="395074666-fc48d433-c0bd-4455-8c05-32311a51aeb7" src="https://github.com/user-attachments/assets/1019067e-2598-4ed7-a926-6d1816562711" />

# RESULT:
   successfully performed Feature Encoding and Transformation process

       
