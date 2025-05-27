import numpy as np
import pandas as pd
import shap

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

def explain_order(model, sample, feature_names, X, X_scaled, Y, x_scaler):
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(sample)
    
    # Get prediction and base value
    prediction = model.predict(sample)[0][0]
    y_binary = (prediction > 0.5).astype(int) 
    base_value = shap_values.base_values[0][0]  # Access the scalar value
    sampleinv = x_scaler.inverse_transform(sample)
    
    # Initialize a list to store output lines
    output_lines = []
    
    # Add basic info to output
    output_lines.append(f"🔍 Return Score: {prediction:.2f}")
    # output_lines.append(f"📊 Ortalama Risk: {base_value:.2f}\n")  # Uncomment if needed
    
    # Get top 5 features
    output_lines.append("💡 Top 5 influential features:")
    for i in np.argsort(-np.abs(shap_values.values[0]))[:5]:
        effect = shap_values.values[0][i]
        if y_binary == 1:
            Xinfo = X[Y == 0]
        else:
            Xinfo = X[Y == 1]

        feature_mean = Xinfo[feature_names[i]].mean()

        if sampleinv[0][i] > feature_mean:
            output_lines.append(f"- Large {feature_names[i]} value ({sampleinv[0][i]:.2f} > mean={feature_mean:.2f}) : {'increases the risk by' if effect > 0 else 'decreases the risk by'} {abs(effect):.2f}")
        else:
            output_lines.append(f"- Small {feature_names[i]} value ({sampleinv[0][i]:.2f} < mean={feature_mean:.2f}) : {'increases the risk by' if effect > 0 else 'decreases the risk by'} {abs(effect):.2f}")
    
    # Join all lines with newlines and return as a single string
    return '\n'.join(output_lines)

"""def explain_order(model, sample, feature_names,X,X_scaled,Y,x_scaler):
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
"""
