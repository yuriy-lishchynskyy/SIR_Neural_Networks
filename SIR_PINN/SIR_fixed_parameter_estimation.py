import numpy as np
import sciann as sn
import matplotlib.pyplot as plt
import scipy as sci

def PrepareData(n_colo):
    data = np.genfromtxt('output_sir_fixed.csv', delimiter=',')

    # select time data, ignoring header row
    t = data[:, 0][1:, ]
    n = len(t)
    t = t.reshape((n, ))

    # create new time data - for every data t, have n_colo no. collocation points
    n_points = ((n - 1) * n_colo) + 1
    t_n = np.linspace(t[0], t[-1], n_points)

    # select sir data
    s = data[:, 1][1:, ].reshape((n, 1))
    i = data[:, 2][1:, ].reshape((n, 1))
    r = data[:, 3][1:, ].reshape((n, 1))

    # interpolate data
    s_i = sci.interpolate.CubicSpline(t, s)
    i_i = sci.interpolate.CubicSpline(t, i)
    r_i = sci.interpolate.CubicSpline(t, r)

    # new collocation points
    s_n = s_i(t_n).reshape((n_points, 1))
    i_n = i_i(t_n).reshape((n_points, 1))
    r_n = r_i(t_n).reshape((n_points, 1))

    # scale to population
    N = 100

    s_n = s_n / N
    i_n = i_n / N
    r_n = r_n / N

    return (t_n, s_n, i_n, r_n)


n_colo = 10
t_train, s_train, i_train, r_train = PrepareData(n_colo)

# define inputs (1 = time)
t = sn.Variable("t", dtype='float64')

# define neural network structure (3 outputs = S / I / R, 4 layers x 32 nodes)
S = sn.Functional("S", [t], 8 * [20], 'tanh')
I = sn.Functional("I", [t], 8 * [20], 'tanh')
R = sn.Functional("R", [t], 8 * [20], 'tanh')

# define 2 SIR parameters to be estimated
param1 = sn.Parameter(np.random.rand(), inputs=[t], name="beta")
param2 = sn.Parameter(np.random.rand(), inputs=[t], name="gamma")

# define ODE derivatives
s_t = sn.diff(S, t)
i_t = sn.diff(I, t)
r_t = sn.diff(R, t)

# Define state constraints
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

model.summary()

# x-values
input_data = [t_train]

# y-values = target values for states (S, I, R)
data_d1 = s_train
data_d2 = i_train
data_d3 = r_train

# y-values = target values for residuals (dS, dI, dR)
data_c1 = 'zeros'
data_c2 = 'zeros'
data_c3 = 'zeros'

target_data = [data_d1, data_d2, data_d3, data_c1, data_c2, data_c3]

history = model.train(
    x_true=input_data,
    y_true=target_data,
    epochs=3000,
    batch_size=16,
    shuffle=False,
    learning_rate=0.001,
    reduce_lr_after=100,
    stop_loss_value=1e-8,
    verbose=2
)

print("beta: {}".format(param1.value))
print("gamma: {}".format(param2.value))

# print training history
plt.figure()
plt.semilogy(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# predict states/derivatives using trained model
sir_output = model.predict(input_data)

# plot states and derivatives
plt.figure()
plt.plot(t_train, sir_output[0], t_train, sir_output[1], t_train, sir_output[2])
plt.xlabel('Day')
plt.ylabel('Value')
plt.title("PINN - States")
plt.legend(["S(t)", "I(t)", "R(t)"])
plt.show()

plt.figure()
plt.plot(t_train, sir_output[3], t_train, sir_output[4], t_train, sir_output[5])
plt.xlabel('Day')
plt.ylabel('Value')
plt.title("PINN - Residuals")
plt.legend(["S(t)", "I(t)", "R(t)"])
plt.show()
