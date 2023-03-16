import csv
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #Q2
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)  

    plt.plot(df['year'], df['days'])
    # naming the x axis
    plt.xlabel('Year')
    # naming the y axis
    plt.ylabel('Number of frozen days')
    plt.savefig("plot.jpg")
    #function to show the plot
    plt.show()

    print("Q3a:") #n × 2 array
    X = np.empty((len(df),2), dtype = int)
    col_1 = ['1'] * len(df)
    col_2 = df['year']
    for row in range(len(df)):
        X[row][0] = col_1[row]
        X[row][1]= col_2[row]
    print(X) 

    print("Q3b:") #nx1 numpy matrix
    Y = np.empty((len(df)), dtype = int)
    for row in range(len(df)):
        Y[row]= df['days'][row]
    print(Y) 

    print("Q3c:") #Compute the matrix product Z = XTX.
    Z = np.dot(np.transpose(X),X)
    print(Z)

    print("Q3d:") #Compute the inverse of XTX.
    I = np.linalg.inv(Z)
    print(I)

    print("Q3e:") #PI = I * XT 
    PI = np.dot(I,np.transpose(X))
    print(PI)

    print("Q3f:") #hat_beta = I * XTY
    hat_beta = np.dot(I,np.dot(np.transpose(X),Y))
    print(hat_beta)

    x_test = 2021
    y_test = hat_beta[0] + hat_beta[1] * x_test
    print("Q4: " + str(y_test))

    if hat_beta[1] > 0:
        print('Q5a: >')
    elif hat_beta[1] == 0:
        print('Q5a: =')
    elif hat_beta[1] < 0:
        print('Q5a: <')

    print('Q5b: The slope coefficient is negative ' 
    + 'because the number of frozen days (y-axis) decreases while the year (x-axis) increases '
    + 'and it leads to the function in a downhill direction.')

    if hat_beta[1] != 0:
        x_star = str((0 - hat_beta[0]) / hat_beta[1])
    print('Q6a: ' + x_star)

    print('Q6b: The value of x∗ is not a compelling prediction ' +
     'because we calculate the dependent variable (x*) ' + 
     'which is considered a result of changes in its corresponding dependent variable (y = 0).')
