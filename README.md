# Fashion Product Classifier

This project implements a **multi-label classification model** to predict key attributes for fashion products:
- **Color** of the product
- **Type of product** (e.g., T-shirt, shoes, etc.)
- **Preferred season** for using the product
- **Gender** (Men, Women, Unisex)

## Steps Followed
1. **Data Cleaning**: Removed missing values and irrelevant columns; filtered out infrequent categories.
2. **EDA (Exploratory Data Analysis)**: Visualized product distributions using bar and pie charts.
3. **Model Building**: Used MobileNetV2 with transfer learning for multi-output classification.
4. **Model Testing**: Predicted attributes for random test images.

## Running the Streamlit App

To run the application locally, follow these steps:

1. **Create a Folder**:
   - Add `main.py` and `requirements.txt` into the folder.

2. **Download Model and Encoded Files**:
   - Download the `.h5` model and encoded `.pkl` files from the following link:  
     [Download the model](https://drive.google.com/drive/folders/1EnbFkoD1PNaEnRJG7ndMklRYarZ4VZV3?usp=sharing)

3. **Install Dependencies**:
   - Install the required dependencies using the following command:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the App**:
   - Start the Streamlit app with the command:
     ```bash
     streamlit run main.py
     ```
