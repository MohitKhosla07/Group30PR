

# import numpy as np
# import pickle
import streamlit as st

#loading the saved model
persistedModel = final_pipeline.load("s3://projectusedcardatset/Saving_model")





columns=['back_legroom', 'city_fuel_economy', 'daysonmarket', 'dealer_zip', 'engine_displacement', 
             'franchise_dealer', 'front_legroom', 'fuel_tank_volume', 'highway_fuel_economy', 
             'maximum_seating', 'mileage', 'owner_count',  'seller_rating', 'torque', 
             'width', 'transmission_id', 'wheelsystem_id', 'fueltype_id ', 'bodytype_id ', 'enginetype_id']



#creating a function for prediction 
def car_prediction(input_data):
    input_data1 = tuple(input_data)
    test = spark.createDataFrame([input_data1], columns)
    prediction = persistedModel.predict(test)
    return prediction


def main():

    #giving #title
    st.title('Welcome to Car Price Prediction')

    st.subheader('This app predicts the price of a car you want to buy.Try filling the details below:')



    #getting the input data from the user
    
    back_legroom = st.text_input("Back leg room ")
    city_fuel_economy = st.text_input("city fuel economy")
    daysonmarket = st.text_input("days on market")
    dealer_zip = st.text_input("dealer zip")
    engine_displacement = st.text_input("engine displacement")
    franchise_dealer = st.text_input("franchise dealer")
    front_legroom = st.text_input("front legroom")
    fuel_tank_volume = st.text_input("fuel tank volume")
#     height = st.text_input("height")
    highway_fuel_economy = st.text_input("highway fuel economy")
    horsepower = st.text_input("horsepower")
#     length = st.text_input("length")
    maximum_seating = st.text_input("maximum seating")
    mileage = st.text_input("mileage")
    owner_count = st.text_input("owner count")
    seller_rating = st.text_input("seller rating")
    torque = st.text_input("torque")
#     wheelbase = st.text_input("wheelbase")
    width = st.text_input("width")
    transmission_id = st.text_input("transmission id")
    wheelsystem_id = st.text_input("wheelsystem id")
    fueltype_id = st.text_input("fueltype id")
    bodytype_id = st.text_input("bodytype id")
    enginetype_id= st.text_input("enginetype id")
#     is_newIndex = st.text_input("is new Index")
#     
    
    #code for prediction

    result = 0

    #creating a button for prediction

    if st.button('Price prediction'):
        result = car_prediction([back_legroom, city_fuel_economy, daysonmarket, dealer_zip,
        engine_displacement, franchise_dealer, front_legroom, fuel_tank_volume, highway_fuel_economy, horsepower,maximum_seating,
        mileage, owner_count, seller_rating, torque, width,  transmission_id, wheelsystem_id,
        fueltype_id , bodytype_id, enginetype_id])

    st.success('Car price is ${}'.format(result))

if __name__ == '__main__':
    main()
