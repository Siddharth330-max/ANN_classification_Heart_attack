# Heart Attack Prediction Project Using Artificial Neural Network(ANN)

This project uses machine learning to predict heart attacks using a cleaned dataset and neural network model, and a Streamlit Web app for live predictions.

---

##  Project Structure (in recommended reading order)

1. [Text_preprocessing.ipynb](./Text_preprocessing.ipynb)  
   - Clean and prepare the dataset for modeling.
2. [ANN.ipynb](./ANN.ipynb) 
   - Build and train the Artificial Neural Network.
3. [predictions.ipynb](./predictions.ipynb)  
   - Make predictions using the trained model.
4. [app.py](./app.py) 
   - Streamlit web app for live prediction deployment.
5. [Cleaned_Heart_attack_dataset.csv](./Cleaned_Heart_attack_dataset.csv)
   - Final cleaned dataset.
6. [Heart_attack_data(csv)](./Heart_attack_data(csv))  
   - Original raw dataset.
7. [model.h5](./model.h5)  
   - Trained Keras model.
8. [scaler.pkl](./scaler.pkl) 
   - Saved feature scaler.
9. [requirements.txt](./requirements.txt)  
   - Required packages for the environment.

---

##  How to Run

```bash
pip install -r requirements.txt
streamlit run app.py


