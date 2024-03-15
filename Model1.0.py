
import pandas
import numpy
from sklearn.model_selection import   train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report


df= pd.read_csv("heart.csv")
Header = df.rows[0]
X=df.iloc[:,:-1]
y=df['HeartDisease']
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.15, random_state=42)

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(x_train,y_train)

y_pred = rd_model.predict(x_test)
accuracy=accuracy_score (y_test, y_pred)
print(f"Model accuracy : {accuracy : .2f}")
