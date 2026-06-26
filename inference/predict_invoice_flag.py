import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'invoice_flagging', 'models', 'predict_flag_invoice.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'invoice_flagging', 'models', 'scaler.pkl')

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [10, 5],
        "invoice_dollars": [18500, 9000],
        "Freight": [200, 100],
        "total_item_quantity": [10, 5],
        "total_item_dollars": [18000, 9000]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)