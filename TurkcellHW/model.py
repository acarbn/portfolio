"""Ürün İade Risk Skoru

Müşterilerin daha önceki siparişlerindeki indirim oranı, ürün miktarı ve harcama miktarına göre bir
 siparişin iade edilme riskini tahmin eden bir derin öğrenme modeli oluştur.

İpucu:
Order Details tablosunda Discount bilgisi var.

(Northwind küçük olduğu için) İade olayını yüksek indirim + düşük harcama gibi bir mantıkla sahte
etiketleyebilirsin.

Ar-Ge Konuları:
Cost-sensitive Learning: İade edilen ürünlerin firmaya maliyeti daha yüksek. Modeli bu durumu daha 
ciddiye alacak şekilde ağırlıklandır.

Explainable AI (XAI): SHAP veya LIME gibi yöntemlerle "Model neden bu siparişi riskli buldu?" 
açıklamasını çıkar."""

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from keras_tuner.tuners import RandomSearch
import joblib


user="####"
password="####"
host="localhost"
port="5432"
database="deneme"
seed=42
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

query="""SELECT * FROM order_details
ORDER BY order_id ASC, product_id ASC"""

df = pd.read_sql(query, engine)
print(df.head(10))

print(df.groupby(['order_id']).min(['discount']))

def feature_eng(order):
    order["min_disc"] = order.groupby("order_id")['discount'].transform('min')
    order["max_disc"] = order.groupby("order_id")['discount'].transform('max')
    order['mean_disc']=order.groupby("order_id")['discount'].transform('mean')
    order['min_quant']=order.groupby("order_id")['quantity'].transform('min')
    order['max_quant']=order.groupby("order_id")['quantity'].transform('max')
    order['mean_quant']=order.groupby("order_id")['quantity'].transform('mean')
    order['count_quant']=order.groupby("order_id")['product_id'].transform('count')
    order['sum_quant']=order.groupby("order_id")['quantity'].transform('sum')
    order['item_total'] = order['unit_price'] * (1 - order['discount']) * order['quantity']

# Then aggregate using transform
    order['total_payment'] = order.groupby("order_id")['item_total'].transform('sum')
    order['min_payment'] = order.groupby("order_id")['item_total'].transform('min')
    order['max_payment'] = order.groupby("order_id")['item_total'].transform('max')
    order['mean_payment'] = order.groupby("order_id")['item_total'].transform('mean')
    order.drop(columns=['product_id', 'unit_price', 'quantity', 'discount','item_total'], inplace=True)
    order = order.drop_duplicates()
    return order


df=feature_eng(df)

def generate_return_labels(orders_df):
    # Initialize return probabilities with base rate
    return_probs = np.full(len(orders_df), 0.15)

    # Scenario 1: Very high discount + Small quantity
    scenario1_mask = (orders_df['max_disc'] > 0.25) & (orders_df['sum_quant'] < 5)
    return_probs[scenario1_mask] = 0.25
    
    # Scenario 2: Medium discount + High quantity
    scenario2_mask = (orders_df['max_disc'].between(0.15, 0.25, inclusive='right')) & (orders_df['sum_quant'] > 20)
    return_probs[scenario2_mask] = 0.10
    
    # Scenario 3: Low total spending + Any discount
    scenario3_mask = (orders_df['total_payment'] < 50) & (orders_df['max_disc'] > 0)
    return_probs[scenario3_mask] = 0.15
    
    # Generate random returns based on probabilities
    random_values = np.random.rand(len(orders_df))
    return_labels = np.where(random_values < return_probs, 1, 0)
    orders_df['return_label']=return_labels
    
    return orders_df

df=generate_return_labels(df)
print(df)
print("Total order: ", df['return_label'].count())
print("Total returned order: ", df['return_label'].sum())
print(f"Return Rate: {df['return_label'].mean() * 100:.2f}%")


X=df.drop(columns=['order_id','return_label'])
X.to_csv('deep_learning/HW/HW2/Xdata.csv', index=False)
Y=df['return_label']
Y.to_csv('deep_learning/HW/HW2/Ydata.csv', index=False)
combined = pd.concat([X, Y], axis=1)
combined.to_csv('deep_learning/HW/HW2/combined_data.csv', index=False)
df2 = pd.read_sql(query, engine)
grouped_df = df2.groupby('order_id').agg(list).reset_index()
combinedoriginal=pd.concat([grouped_df, Y.reset_index()], axis=1)
combinedoriginal.to_csv('deep_learning/HW/HW2/combined_originaldata.csv', index=False)
# 1. First split WITH stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=seed,
    stratify=Y  # Critical for imbalanced data
)

over = SMOTE(random_state=seed)
X_train_res, y_train_res = over.fit_resample(X_train, y_train)

# Scaling
x_scaler = StandardScaler()
X_tr_scaled = x_scaler.fit_transform(X_train_res)
X_test_scaled = x_scaler.transform(X_test)  # Important: use transform NOT fit_transform
"""
## Weights ###########
f1val=[]
weightsi=np.arange(1, 6, 0.2)
for i in weightsi:
    
    class_weights={0: 1, 1: i}
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_tr_scaled.shape[1],)),  # Explicit input layer
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
    ])


    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",  # NOT MSE for classification
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    model.fit(X_tr_scaled, y_train_res, epochs=500, verbose=0, class_weight=class_weight_dict)
    y_pred2=(model.predict(X_test_scaled) > 0.5).astype(int) 
    cl=classification_report(y_test, y_pred2,output_dict=True)
    f1val.append(cl["1"]["f1-score"])
    del model
max_f1 = max(f1val)
max_index = f1val.index(max_f1)
print(weightsi[max_index])
class_weights={0: 1, 1: float(weightsi[max_index])}
print(class_weights)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
with open("parameters.txt", "a") as f:
    f.write(f"class_weight: {class_weights}")
############################################################################################################################

def build_model(hp):
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Input(shape=(X_tr_scaled.shape[1],)))
    
    # Tune number of units
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units', 2, 64, step=2),
        activation=hp.Choice('activation', ['relu', 'tanh']),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
    ))
    
    # Tune dropout rate
    model.add(tf.keras.layers.Dropout(
        hp.Float('dropout', 0.1, 0.5)
    ))
    
    # Final output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Tune learning rate
    optimizer = tf.keras.optimizers.Adam(
        hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# 4. Configure tuner with class weights
tuner = RandomSearch(
    build_model,
    objective='val_auc',
    max_trials=10,
    executions_per_trial=2,
    directory=None,
    project_name=None
)

# 5. Run search 
tuner.search(
    X_tr_scaled, 
    y_train_res,
    epochs=500,
    validation_data=(X_test_scaled, y_test),
    class_weight=class_weight_dict,  # Pass weights here
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=0
        )
    ]
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Get the best model (with trained weights)
best_model = tuner.get_best_models(num_models=1)[0]

# Print all parameters
print("=== Best Hyperparameters ===")
for param, value in best_hps.values.items():
    print(f"{param:15}: {value}")

# Evaluate best model on test set
print("\n=== Best Model Evaluation ===")
test_loss, test_acc, test_auc = best_model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test AUC: {test_auc:.3f}")
y_probs2 = best_model.predict(X_test_scaled)
y_pred2=(y_probs2 > 0.42345).astype(int) 
print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

# Get predicted probabilities

# Find threshold that maximizes F1-score for class 1
precision, recall, thresholds = precision_recall_curve(y_test, y_probs2)
f1_scores = 2*(precision*recall)/(precision+recall+1e-9)
optimal_idx = np.argmax(f1_scores)
max_threshold = thresholds[optimal_idx]
optimal_threshold = 0.5 * (0.5 + max_threshold)
print(optimal_threshold)
with open("parameters.txt", "a") as f:
    f.write(f"optimal_threshold: {optimal_threshold}")

# Apply new threshold
y_pred_optimized = (y_probs2 >= optimal_threshold).astype(int)
print(confusion_matrix(y_test, y_pred_optimized))
print(classification_report(y_test, y_pred_optimized))

# train on whole dataset, predict for example, explain order



# 1. Save the best model (TensorFlow format)
best_model.save('deep_learning/HW/HW2/return_risk_model.keras')  # or .keras for newer versions

# 2. Save the scaler using joblib
joblib.dump(x_scaler, 'deep_learning/HW/HW2/scaler.joblib')

# 3. Save the optimal threshold if you have one
joblib.dump(optimal_threshold, 'deep_learning/HW/HW2/threshold.joblib')"""



## whole dataset
tf.config.run_functions_eagerly(True)  # Enable eager execution

X_scaled = x_scaler.transform(X)
#X_scaled.to_csv('deep_learning/HW/HW2/Xscaleddata.csv', index=False)

over = SMOTE(random_state=seed)
X_res, y_res = over.fit_resample(X_scaled, Y)
X_res = np.array(X_res, dtype='float32')  # Note the dtype
y_res = np.array(y_res, dtype='float32')
# Scaling
model = tf.keras.models.load_model('deep_learning/HW/HW2/return_risk_model.h5')
# Continue training on full dataset

# If still getting optimizer errors:
model.compile(
    optimizer=tf.keras.optimizers.Adam(),  # Match original type
    loss=model.loss
)
model.set_weights(tf.keras.models.load_model('deep_learning/HW/HW2/return_risk_model.h5').get_weights())
model.fit(X_res, y_res,
          epochs=500)
model.save('deep_learning/HW/HW2/fullmodel.keras') 


"""order=pd.DataFrame({
    "order_id": 10248,
    "product_id": [11,42,72],
    "unit_price":[14,9.8,34.8],
    "quantity":[12,10,5],
    "discount":[0,0,0]
})"""

order=pd.DataFrame({
    "order_id": 10254,
    "product_id": [24,55,74],
    "unit_price":[3.6,19.2,8],
    "quantity":[15,21,21],
    "discount":[0.15,0.15,0]
})

order["min_disc"] = order.groupby("order_id")['discount'].transform('min')
print(order)

order=feature_eng(order)
order=generate_return_labels(order)
orderX=order.drop(columns=['order_id','return_label'])
orderX_scaled = x_scaler.transform(orderX)
orderY=model.predict(orderX_scaled)
print(orderY)  # Probabilities (0 to 1)
y_pred_binary = (orderY > 0.5).astype(int)  # Convert to 0 or 1
print(y_pred_binary)
###################################################################################
def explain_order(model, sample, feature_names,X,X_scaled,Y):
    # Create SHAP explainer
    
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(sample)
    
    # Get prediction and base value
    prediction = model.predict(sample)[0][0]
    y_binary = (prediction > 0.5).astype(int) 
    base_value = shap_values.base_values[0][0]  # Access the scalar value
    sampleinv=x_scaler.inverse_transform(sample)
    # Print basic info
    print(f"🔍 Return Score: {prediction:.2f}")
    #print(f"📊 Ortalama Risk: {base_value:.2f}\n")
    
    # Get 
    print("💡 Top 5 influential features:")
    for i in np.argsort(-np.abs(shap_values.values[0]))[:5]:
        effect = shap_values.values[0][i]
        if y_binary==1:
            Xinfo=X[Y==0]
        else:
            Xinfo=X[Y==1]

        feature_mean= Xinfo[feature_names[i]].mean()

        if sampleinv[0][i]>feature_mean:
            print(f"- Large {feature_names[i]} value ({sampleinv[0][i]:.2f} > mean={feature_mean:.2f}) : {'increases the risk by' if effect > 0 else 'decreases the risk by'} {abs(effect):.2f}")
        else:
            print(f"- Small {feature_names[i]} value ({sampleinv[0][i]:.2f} < mean={feature_mean:.2f}) : {'increases the risk by' if effect > 0 else 'decreases the risk by'} {abs(effect):.2f}")


# Usage
explain_order(model, orderX_scaled, X.columns,X,X_scaled,Y)


