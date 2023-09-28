import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import  RobustScaler
import warnings
import joblib

pd.set_option("display.max_columns",None)
pd.set_option("display.width", 500)
warnings.simplefilter(action="ignore")


df1 = pd.read_csv("hafta7(machine_learning)/TelcoChurn/Telco-Customer-Churn.csv")
df = df1.copy()

def check_data(dataframe,head = 5):
    print("###########################SHAPE########################")
    print(dataframe.shape)
    print("###########################TYPES########################")
    print(dataframe.dtypes)
    print("###########################HEAD########################")
    print(dataframe.head(head))
    print("###########################NULL########################")
    print(dataframe.isnull().sum())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

check_data(df)
df["Churn"] = df["Churn"].apply(lambda x : 1 if x =="Yes" else 0)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")  # satır
    print(f"Variables: {dataframe.shape[1]}")  # değişken
    print(f'cat_cols: {len(cat_cols)}')  # kategorik degişken
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col,plot=True)

def num_summary(dataframe,colname,plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[colname].describe(quantiles))
    if plot:
        if plot == "hist":
            plt.hist(df[colname],bins=20)
            plt.title(colname)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show(block=True)
        elif plot == "scatter" :
            df[colname].plot(kind='density')
            plt.title(f'Density Plot of Numerical Column {colname}')
            plt.xlabel('Value')
            plt.show(block=True)
        else :
            print("Invalid plot type. Use 'hist' for histogram or 'scatter' for scatter plot.")

for col in num_cols:
    num_summary(df,col,plot="scatter")


def target_summ_with_num(dataframe,numcol,target):
    print(dataframe.groupby(target).agg({numcol:"mean"}),end="\n\n\n")


for col in num_cols:
    target_summ_with_num(df,col,"Churn")

def target_summ_with_cat(dataframe,catcol,target):
    print(catcol)
    print(pd.DataFrame({"Target_mean": dataframe.groupby(catcol)[target].mean(),
                        "Ratio" : 100* dataframe[catcol].value_counts() / len(dataframe),
                        "Count" :  dataframe[catcol].value_counts()}), end="\n\n\n")


for col in cat_cols :
    if col == "Churn":
        continue
    target_summ_with_cat(df,col,"Churn")

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


## Eksik değer Analizi

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_col=missing_values_table(df,na_name=True)



def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0) # Bu boş değerlere 1 diğerlerine 0 yaz.

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df,"Churn",na_col)

## Eksik değerleri aylık değerle doldurduk.
df.loc[df["TotalCharges"].isnull(),"TotalCharges"] = df["MonthlyCharges"]

df["tenure"] = df["tenure"] + 1

df[df["tenure"]==1]

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe,col_name):
    low,up = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name] < low )| (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col,check_outlier(df,col))
    if check_outlier(df,col):
        replace_with_thresholds(df,col)


### Base model Kurulumu

dff = df.copy()
dff.head()

cat_cols = [col for col in cat_cols if col not in "Churn"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(df,cat_cols,drop_first=True)

X = dff.drop(["Churn","customerID"],axis=1)
y = dff["Churn"]

## Models

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]


for name,model in models:
    cv_results = cross_validate(model,X,y,cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")



##### Özellik Çıkarımı
# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12 ), "NEW_TENURE_YEAR"] = "0-1 YEAR"
df.loc[(df["tenure"] >= 13) & (df["tenure"] <= 24) , "NEW_TENURE_YEAR"] = "1-2 YEAR"
df.loc[(df["tenure"] >= 25) & (df["tenure"] <= 36) , "NEW_TENURE_YEAR"] = "2-3 YEAR"
df.loc[(df["tenure"] >= 37) & (df["tenure"] <= 48 ), "NEW_TENURE_YEAR"] = "3-4 YEAR"
df.loc[(df["tenure"] >= 49) & (df["tenure"] <= 60 ), "NEW_TENURE_YEAR"] = "4-5 YEAR"
df.loc[(df["tenure"] >= 61) & (df["tenure"] <= 72) , "NEW_TENURE_YEAR"] = "4-6 YEAR"

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_ENGAGED"] = df["Contract"].apply(lambda x: 1 if  x in ["One year", "Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_NOPROT"] = df.apply(lambda x:  1 if (x["OnlineBackup"] != "Yes") or ( x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0,axis=1 )

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_ENGAGED"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)


# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / df["tenure"]

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_col = [col for col in cat_cols if df[col].dtype == 'O' and df[col].nunique() ==2]

for col in binary_col:
    label_encoder(df,col)


cat_cols = [col for col in cat_cols if col not in binary_col and col not in ["Churn", "NEW_TotalServices"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)


X = df.drop(["Churn","customerID"], axis=1)
y = df["Churn"]


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name,model in models:
    cv_results = cross_validate(model,X,y,cv=10,n_jobs=-1,scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########################{name}########################")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

################################################
# Random Forests
################################################


rf_model = RandomForestClassifier(random_state=17)


rf_params = {"max_depth": [5, 8, None], # Ağacın maksimum derinliği, kac seviye bölünmeli
             "max_features": [3, 5, 7, "auto"], # En iyi bölünmeyi ararken göz önünde bulundurulması gereken özelliklerin sayısı
             "min_samples_split": [2, 5, 8, 15, 20], # Bir node'u bölmek için gereken minimum örnek sayısı, en son kac gözlem kalmalı
             "n_estimators": [100, 200, 500]} # Ağaç sayısı

rf_best_grid = GridSearchCV(rf_model,rf_params,cv = 5, verbose = True,n_jobs=-1).fit(X,y)

rf_best_grid.best_params_

rf_final = RandomForestClassifier(**rf_best_grid.best_params_,random_state=17).fit(X,y)
cv_results = cross_validate(rf_final,X,y,cv=5,n_jobs=-1,scoring=["accuracy", "f1","recall","precision"])

joblib.dump(rf_final, "rf_final.pkl")


model = joblib.load("rf_final.pkl")

random_user = X.sample(1, random_state=45)
model.predict(random_user)
