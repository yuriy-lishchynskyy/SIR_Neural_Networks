import numpy as np
import sciann as sn
import matplotlib.pyplot as plt
import scipy as sci

print("Packages Loaded")

def PrepareDataSIR(window, points, n_colo):
    data = np.genfromtxt('output_sird_beta_expo_small.csv', delimiter=',')

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
    d_d = data[:, 4][1:points + 2, ].reshape((points + 1, 1)) / N0

    beta_real = data[:, 6][1:points + 2, ]
    gamma_real = data[:, 7][1:points + 2, ]
    mu_real = data[:, 8][1:points + 2, ]

    time_real = t_d

    # set up arrays of predicted parameters (add leading 0's due to window)
    beta_pred = []
    gamma_pred = []
    mu_pred = []

    # interpolate data
    s_i = sci.interpolate.CubicSpline(t_d, s_d)
    i_i = sci.interpolate.CubicSpline(t_d, i_d)
    r_i = sci.interpolate.CubicSpline(t_d, r_d)
    d_i = sci.interpolate.CubicSpline(t_d, d_d)

    # new collocation points
    s_n = s_i(t_n).reshape((n_points, 1))
    i_n = i_i(t_n).reshape((n_points, 1))
    r_n = r_i(t_n).reshape((n_points, 1))
    d_n = d_i(t_n).reshape((n_points, 1))

    # array of n_series windows, each with "window" no. of elements
    t_arr = []
    s_arr = []
    i_arr = []
    r_arr = []
    d_arr = []

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
        d_w = d_n[k_start:k_start + n_window, ].reshape((n_window, 1))

        # add interpolation to list
        t_arr.append(t_w)
        s_arr.append(s_w)
        i_arr.append(i_w)
        r_arr.append(r_w)
        d_arr.append(d_w)

        time_pred.append(window + k)

        # starting index for next day
        k_start = (k + 1) * n_colo

    return (t_arr, s_arr, i_arr, r_arr, d_arr, time_real, beta_real, gamma_real, mu_real, time_pred, beta_pred, gamma_pred, mu_pred)


# read in and prepare windowed data
window = 5 # number of previous days to fit for given day
points = 35 # number of days of data to use (from start)
collocation = 10 # no. of additional points to interpolate (per actual data point)
t_trains, s_trains, i_trains, r_trains, d_trains, t_true, beta_true, gamma_true, mu_true, t_pred, beta_pred, gamma_pred, mu_pred = PrepareDataSIR(window, points, collocation)

print("Data Loaded")

for i in range(0, len(beta_true)):
    print("{0} - {1} - {2}".format(beta_true[i], gamma_true[i], mu_true[i]))

# define inputs (1 = time)
t = sn.Variable("t", dtype='float64')

# define neural network structure (3 outputs = S / I / R, 8 layers x 20 nodes)
S = sn.Functional("S", [t], 8 * [20], 'tanh')
I = sn.Functional("I", [t], 8 * [20], 'tanh')
R = sn.Functional("R", [t], 8 * [20], 'tanh')
D = sn.Functional("D", [t], 8 * [20], 'tanh')

# define SIR parameters to be estimated
param1 = sn.Parameter(np.random.rand(), inputs=[t], name="beta")
param2 = sn.Parameter(np.random.rand(), inputs=[t], name="gamma")
param3 = sn.Parameter(np.random.rand(), inputs=[t], name="mu")

# define ODE derivatives
s_t = sn.diff(S, t)
i_t = sn.diff(I, t)
r_t = sn.diff(R, t)
d_t = sn.diff(D, t)

# Define model state constraints
d1 = sn.Data(S)
d2 = sn.Data(I)
d3 = sn.Data(R)
d4 = sn.Data(D)

# define residual constraints
c1 = sn.Tie(s_t, -param1 * S * I)
c2 = sn.Tie(i_t, (param1 * S * I) - (param2 * I) - (param3 * I))
c3 = sn.Tie(r_t, (param2 * I))
c4 = sn.Tie(d_t, (param3 * I))

# Define the optimization model (set of inputs and constraints)
model = sn.SciModel(
    inputs=[t],
    targets=[d1, d2, d3, d4, c1, c2, c3, c4],
    loss_func="mse",
    optimizer='adam'
)

n_runs = len(t_trains)

print("Number of Training Windows: {0}".format(n_runs))

for i in range(0, n_runs):

    print("Iteration: {0}".format(i + 1))

    # extract data for current step window
    t_train = t_trains[i]
    s_train = s_trains[i]
    i_train = i_trains[i]
    r_train = r_trains[i]
    d_train = d_trains[i]

    # x-values
    input_data = [t_train]

    # y-values = actual S/I/R values
    data_d1 = s_train
    data_d2 = i_train
    data_d3 = r_train
    data_d4 = d_train

    # y-values = target values for residuals (dS, dI, dR)
    data_c1 = 'zeros'
    data_c2 = 'zeros'
    data_c3 = 'zeros'
    data_c4 = 'zeros'

    target_data = [data_d1, data_d2, data_d3, data_d4, data_c1, data_c2, data_c3, data_c4]

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
    mu_pred.append(param3.value[0])

# print out time-varying parameters
print("\nt \t\t beta \t\t gamma \t\t mu")

for i in range(0, len(t_pred)):
    print("{0} \t\t {1:.4f} \t {2:.4f} \t {3:.4f}".format(t_pred[i], beta_pred[i], gamma_pred[i], mu_pred[i]))

# plot parameters
plt.figure()
plt.plot(t_pred, beta_pred, 'b-', label='Beta - Estimated')
plt.plot(t_true, beta_true, 'bo', label='Beta - Actual')
plt.plot(t_pred, gamma_pred, 'r-', label='Gamma - Estimated')
plt.plot(t_true, gamma_true, 'ro', label='Gamma - Actual')
plt.plot(t_pred, mu_pred, 'g-', label='Mu - Estimated')
plt.plot(t_true, mu_true, 'go', label='Mu - Actual')
plt.xticks(np.arange(min(t_true), max(t_true) + 1, 1.0))
plt.yticks(np.arange(0, 0.55, 0.05))
plt.xlabel("Day")
plt.ylabel("Value")
plt.legend()
plt.show()
