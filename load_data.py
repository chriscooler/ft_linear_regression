import csv


def load_data(file_path):
    """
    Reads mileage and price data from a CSV file.
    
    Parameters:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - tuple: Two lists, one with mileage data and another with price data.
    """
    mileage = []
    price = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # skip header
        for row in csv_reader:
            mileage.append(float(row[0]))
            price.append(float(row[1]))
    # print("KM: ", mileage)
    # print("price: ", price)
    return mileage, price


def compute_cost(mileage, price, theta0, theta1):
    """
    Computes the cost function for linear regression.
    
    Parameters:
    - mileage (list of float): The input feature (mileage) data.
    - price (list of float): The target variable (price) data.
    - theta0 (float): The intercept term (theta0) for the model.
    - theta1 (float): The slope term (theta1) for the model.
    
    Returns:
    - float: The computed cost value.
    """
    m = len(price)
    total_cost = 0
    for i in range(m):
        prediction = theta0 + theta1 * mileage[i]
        total_cost += (prediction - price[i]) ** 2
    return total_cost / (2 * m)


def gradient_descent(mileage, price, theta0, theta1, alpha, iterations):
    """
    Performs gradient descent to learn the optimal parameters for linear regression.
    
    Parameters:
    - mileage (list of float): The input feature (mileage) data.
    - price (list of float): The target variable (price) data.
    - theta0 (float): The initial intercept term (theta0).
    - theta1 (float): The initial slope term (theta1).
    - alpha (float): The learning rate for gradient descent.
    - iterations (int): The number of iterations to run the gradient descent.
    
    Returns:
    - tuple: The updated values for theta0 and theta1 after training.
    """
    m = len(price)
    
    for _ in range(iterations):
        sum_error_theta0 = 0
        sum_error_theta1 = 0
        
        # Calculate the sum of errors for theta0 and theta1
        for i in range(m):
            prediction = theta0 + theta1 * mileage[i]
            error = prediction - price[i]
            sum_error_theta0 += error
            sum_error_theta1 += error * mileage[i]
        
        # Update thetas
        theta0 = theta0 - alpha * (sum_error_theta0 / m)
        theta1 = theta1 - alpha * (sum_error_theta1 / m)
    
    return theta0, theta1


def predict_price(mileage, theta0, theta1):
    """
    Predicts the price of a car given its mileage using learned parameters.
    
    Parameters:
    - mileage (float): The mileage of the car.
    - theta0 (float): The learned intercept term (theta0).
    - theta1 (float): The learned slope term (theta1).
    
    Returns:
    - float: The predicted price of the car.
    """
    return theta0 + theta1 * mileage


def main():
    # Load data
    mileage, price = load_data('data.csv')
    
    # Initialize parameters
    theta0 = 0
    theta1 = 0
    alpha = 0.00001  # Learning rate (you can tweak this)
    iterations = 10000  # Number of iterations (you can adjust this)

    # Perform gradient descent to get optimal thetas
    theta0, theta1 = gradient_descent(mileage, price, theta0, theta1,
                                      alpha, iterations)
    
    # Prediction phase
    user_mileage = float(input("Enter the mileage of the car: "))
    estimated_price = predict_price(user_mileage, theta0, theta1)
    print(f"Estimated price for mileage {user_mileage}: \
             ${estimated_price:.2f}")


if __name__ == "__main__":
    main()

