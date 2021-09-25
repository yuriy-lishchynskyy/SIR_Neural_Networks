import numpy as np
import sciann as sn
import matplotlib.pyplot as plt
import scipy as sci

print("Packages Loaded")

def PrepareDataSIR(window, points, n_colo):
    data = np.genfromtxt('output_sir_beta_expo_large.csv', delimiter=',')

    # select time data, ignoring header row
    t_d = data[:points + 2, 0][1:, ]
    n = len(t_d)

    # create new time data - for every data t, have n_colo no. collocation points
    n_points = ((n - 1) * n_colo) + 1
    t_n = np.linspace(t_d[0], t_d[-1], n_points)

    n_window = window * n_colo # size of window after interpolation
    n_series = points - window # no. of windows to be created

    N0 = 100

    # select non-interpolated sir data
    s_d = data[:, 1][1:points + 2, ].reshape((points + 1, 1)) / N0
    i_d = data[:, 2][1:points + 2, ].reshape((points + 1, 1)) / N0
    r_d = data[:, 3][1:points + 2, ].reshape((points + 1, 1)) / N0

    beta_real = data[:, 5][1:points + 2, ]
    gamma_real = data[:, 6][1:points + 2, ]

    time_real = t_d

    # set up arrays of predicted parameters (add leading 0's due to window)
    beta_pred = []
    gamma_pred = []

    # interpolate data
    s_i = sci.interpolate.CubicSpline(t_d, s_d)
    i_i = sci.interpolate.CubicSpline(t_d, i_d)
    r_i = sci.interpolate.CubicSpline(t_d, r_d)

    # new collocation points
    s_n = s_i(t_n).reshape((n_points, 1))
    i_n = i_i(t_n).reshape((n_points, 1))
    r_n = r_i(t_n).reshape((n_points, 1))

    # array of n_series windows, each with "window" no. of elements
    t_arr = []
    s_arr = []
    i_arr = []
    r_arr = []

    # starting day (every "window" no. of points)
    k_start = 0

    # time for predictions
    time_pred = []

    for k in range(0, n_series + 1):

        # select t and sir data window
        t_w = t_n[k_start:k_start + n_window, ].reshape((n_window, 1))
        s_w = s_n[k_start:k_start + n_window, ].reshape((n_window, 1))
        i_w = i_n[k_start:k_start + n_window, ].reshape((n_window, 1))
        r_w = r_n[k_start:k_start + n_window, ].reshape((n_window, 1))

        # add interpolation to list
        t_arr.append(t_w)
        s_arr.append(s_w)
        i_arr.append(i_w)
        r_arr.append(r_w)

        time_pred.append(window + k)

        # starting index for next day
        k_start = (k + 1) * n_colo

    return (t_arr, s_arr, i_arr, r_arr, time_real, beta_real, gamma_real, time_pred, beta_pred, gamma_pred)


# read in and prepare windowed data
window = 5 # number of previous days to fit for given day
points = 35 # number of days of data to use (from start)
collocation = 10 # no. of additional points to interpolate (per actual data point)
t_trains, s_trains, i_trains, r_trains, t_true, beta_true, gamma_true, t_pred, beta_pred, gamma_pred = PrepareDataSIR(window, points, collocation)

print("Data Loaded")

for i in range(0, len(beta_true)):
    print("{0} - {1}".format(beta_true[i], gamma_true[i]))

# define inputs (1 = time)
t = sn.Variable("t", dtype='float64')

# define neural network structure (3 outputs = S / I / R, 8 layers x 20 nodes)
S = sn.Functional("S", [t], 8 * [20], 'tanh')
I = sn.Functional("I", [t], 8 * [20], 'tanh')
R = sn.Functional("R", [t], 8 * [20], 'tanh')

# define SIR parameters to be estimated
param1 = sn.Parameter(np.random.rand(), inputs=[t], name="beta")
param2 = sn.Parameter(np.random.rand(), inputs=[t], name="gamma")

# define ODE derivatives
s_t = sn.diff(S, t)
i_t = sn.diff(I, t)
r_t = sn.diff(R, t)

# Define model state constraints
d1 = sn.Data(S)
d2 = sn.Data(I)
d3 = sn.Data(R)

# define residual constraints
c1 = sn.Tie(s_t, -param1 * S * I)
c2 = sn.Tie(i_t, (param1 * S * I) - (param2 * I))
c3 = sn.Tie(r_t, (param2 * I))

# Define the optimization model (set of inputs and constraints)
model = sn.SciModel(
    inputs=[t],
    targets=[d1, d2, d3, c1, c2, c3],
    loss_func="mse",
    optimizer='adam'
)

n_runs = len(t_trains)
# n_runs = 10

print("Number of Training Windows: {0}".format(n_runs))

for i in range(0, n_runs):

    print("Iteration: {0}".format(i + 1))

    # extract data for current step window
    t_train = t_trains[i]
    s_train = s_trains[i]
    i_train = i_trains[i]
    r_train = r_trains[i]

    # x-values
    input_data = [t_train]

    # y-values = actual S/I/R values
    data_d1 = s_train
    data_d2 = i_train
    data_d3 = r_train

    # y-values = target values for residuals (dS, dI, dR)
    data_c1 = 'zeros'
    data_c2 = 'zeros'
    data_c3 = 'zeros'

    target_data = [data_d1, data_d2, data_d3, data_c1, data_c2, data_c3]

    # fit model
    history = model.train(
        x_true=input_data,
        y_true=target_data,
        epochs=1500,
        batch_size=4,
        shuffle=False,
        learning_rate=0.001,
        reduce_lr_after=500,
        stop_loss_value=1e-10,
        verbose=0
    )

    # store estimated parameters
    beta_pred.append(param1.value[0])
    gamma_pred.append(param2.value[0])

# print out time-varying parameters
print("\nt \t\t beta \t\t gamma")

for i in range(0, len(t_pred)):
    print("{0} \t\t {1:.4f} \t {2:.4f}".format(t_pred[i], beta_pred[i], gamma_pred[i]))

# plot parameters
plt.figure()
plt.plot(t_pred, beta_pred, 'b-', label='Beta - Estimated')
plt.plot(t_true, beta_true, 'bo', label='Beta - Actual')
plt.plot(t_pred, gamma_pred, 'r-', label='Gamma - Estimated')
plt.plot(t_true, gamma_true, 'ro', label='Gamma - Actual')
plt.xticks(np.arange(min(t_true), max(t_true) + 1, 1.0))
plt.yticks(np.arange(0, 0.55, 0.05))
plt.xlabel("Day")
plt.ylabel("Value")
plt.legend()
plt.show()
