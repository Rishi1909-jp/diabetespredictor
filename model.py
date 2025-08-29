import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load dataset (for demo)
data = load_diabetes()
X, y = data.data, data.target

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train simple model
model = LinearRegression()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "diabetes_model.pkl")

print("✅ Model trained & saved as diabetes_model.pkl")
