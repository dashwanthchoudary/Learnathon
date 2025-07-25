import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model and sample data
model = joblib.load('insurance_fraud_model.pkl')
sample_data = pd.read_csv('Auto_Insurance_Fraud_Claims_File01.csv')  # Your claims data

# Get list of all columns the model expects
expected_columns = [
    'Customer_Life_Value1', 'Age_Insured', 'Policy_Ded', 'Policy_Premium',
    'Umbrella_Limit', 'Insured_Zip', 'Capital_Gains', 'Capital_Loss',
    'Accident_Hour', 'Num_of_Vehicles_Involved', 'Bodily_Injuries',
    'Witnesses', 'Auto_Year', 'Vehicle_Cost', 'Annual_Mileage',
    'DiffIN_Mileage', 'Low_Mileage_Discount', 'Commute_Discount',
    'Total_Claim', 'Injury_Claim', 'Property_Claim', 'Vehicle_Claim',
    'Vehicle_Age', 'Injury_to_Total_Ratio', 'Property_to_Total_Ratio',
    'Vehicle_to_Total_Ratio', 'Policy_State', 'Policy_BI', 'Gender',
    'Education', 'Occupation', 'Hobbies', 'Insured_Relationship',
    'Garage_Location', 'Accident_Date', 'Accident_Type', 'Collision_Type',
    'Accident_Severity', 'authorities_contacted', 'Acccident_State',
    'Acccident_City', 'Accident_Location', 'Property_Damage',
    'Police_Report', 'Auto_Make', 'Auto_Model', 'Vehicle_Color'
]

def prepare_claim_data(claim_row):
    """Prepare a claim record for prediction by ensuring all expected columns exist"""
    # Create a DataFrame with all expected columns
    prediction_data = pd.DataFrame(columns=expected_columns)
    
    # Fill with default values first
    prediction_data.loc[0] = 0  # For numerical columns
    for col in prediction_data.select_dtypes(include=['object']).columns:
        prediction_data[col] = 'missing'
    
    # Map available data from the claim record
    for col in claim_row.index:
        if col in prediction_data.columns:
            prediction_data[col] = claim_row[col]
    
    # Calculate derived features (same as in training)
    if 'Total_Claim' in prediction_data.columns and 'Vehicle_Cost' in prediction_data.columns:
        prediction_data['Vehicle_to_Total_Ratio'] = (
            prediction_data['Vehicle_Cost'] / prediction_data['Total_Claim'] 
            if prediction_data['Total_Claim'].iloc[0] > 0 else 0
        )
    
    if 'Injury_Claim' in prediction_data.columns and 'Total_Claim' in prediction_data.columns:
        prediction_data['Injury_to_Total_Ratio'] = (
            prediction_data['Injury_Claim'] / prediction_data['Total_Claim'] 
            if prediction_data['Total_Claim'].iloc[0] > 0 else 0
        )
    
    if 'Property_Claim' in prediction_data.columns and 'Total_Claim' in prediction_data.columns:
        prediction_data['Property_to_Total_Ratio'] = (
            prediction_data['Property_Claim'] / prediction_data['Total_Claim'] 
            if prediction_data['Total_Claim'].iloc[0] > 0 else 0
        )
    
    if 'Auto_Year' in prediction_data.columns:
        prediction_data['Vehicle_Age'] = datetime.now().year - prediction_data['Auto_Year']
    
    if 'Policy_State' in prediction_data.columns and 'Acccident_State' in prediction_data.columns:
        prediction_data['State_Mismatch'] = (
            1 if prediction_data['Policy_State'].iloc[0] != prediction_data['Acccident_State'].iloc[0] else 0
        )
    
    return prediction_data

# Streamlit app
st.title("Insurance Fraud Detection System")

# Option 1: Search by Claim ID
claim_id = st.text_input("Enter Claim ID:")

if claim_id:
    try:
        # Find claim in database
        claim_row = sample_data[sample_data['Claim_ID'] == claim_id].iloc[0]
        
        # Prepare data for model
        prediction_data = prepare_claim_data(claim_row)
        
        # Make prediction
        fraud_prob = model.predict_proba(prediction_data)[0][1]
        
        # Display results
        st.subheader(f"Results for Claim {claim_id}")
        st.metric("Fraud Probability", f"{fraud_prob:.1%}")
        
        if fraud_prob > 0.5:
            st.error("🚨 High fraud risk detected!")
            st.write("Recommended action: This claim requires manual review")
        else:
            st.success("✅ Low fraud risk")
            st.write("Recommended action: Standard processing")
        
        # Show key details
        st.subheader("Claim Summary")
        cols = st.columns(4)
        cols[0].metric("Total Claim", f"${claim_row['Total_Claim']:,.2f}")
        cols[1].metric("Vehicle Value", f"${claim_row['Vehicle_Cost']:,.2f}")
        cols[2].metric("Injury Claim", f"${claim_row['Injury_Claim']:,.2f}")
        cols[3].metric("Property Claim", f"${claim_row['Property_Claim']:,.2f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Automobile Model:** {claim_row.get('Auto_Model', 'Unknown')}, {claim_row.get('Auto_Make', 'Unknown')}")
            st.write(f"**Automobile Year:** {claim_row.get('Auto_Year', 'Unknown')}")

        with col2:
            st.write(f"**Policy State:** {claim_row.get('Policy_State', 'Unknown')}")
            st.write(f"**Location:** {claim_row.get('Acccident_City', 'Unknown')}, {claim_row.get('Acccident_State', 'Unknown')}")


        with col3:
            st.write(f"**Accident Type:** {claim_row.get('Accident_Type', 'Unknown')}")            
        # st.write(f"**Automobile Model:** {claim_row.get('Auto_Model', 'Unknown')}")
        # st.write(f"**Automobile Year:** {claim_row.get('Auto_Year', 'Unknown')}")
        # st.write(f"**Policy State:** {claim_row.get('Policy_State', 'Unknown')}")
        # st.write(f"**Accident Type:** {claim_row.get('Accident_Type', 'Unknown')}")
        # st.write(f"**Location:** {claim_row.get('Acccident_City', 'Unknown')}, {claim_row.get('Acccident_State', 'Unknown')}")
        
    except IndexError:
        st.error("Claim ID not found in database")
    except Exception as e:
        st.error(f"Error processing claim: {str(e)}")
        st.write("Please ensure your data contains all required columns")

# Add information about required fields
st.sidebar.markdown("""
### Required Claim Data Fields
For this system to work, your claims data must include these key fields:
- Total_Claim
- Vehicle_Cost
- Injury_Claim
- Property_Claim
- Auto_Year
- Policy_State
- Acccident_State
- Accident_Type
- And all other features used in model training
""")