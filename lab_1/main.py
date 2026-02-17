import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, lagrange


X_DATA = [i * 0.1 for i in range(1, 21)]
Y_DATA = [0.17, 0.07, 0.17, 0.05, 0.12, 0.00, 0.01, -0.05, -0.21, -0.50,
          -0.50, -0.86, -1.24, -1.47, -1.79, -2.25, -2.55, -3.18, -3.60, -3.93]


def lagrange_interp(x, y, xq):
    n = len(x)
    s = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if j != i:
                p *= (xq - x[j]) / (x[i] - x[j])
        s += y[i] * p
    return s


def run_through(a, b, c, d):
    n = len(b)
    u = [0] * n
    v = [0] * n
    u[0] = -c[0] / b[0]
    v[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * u[i - 1]
        if i < n - 1:
            u[i] = -c[i] / denom
        v[i] = (d[i] - a[i] * v[i - 1]) / denom
    res_x = [0] * n
    res_x[-1] = v[-1]
    for i in range(n - 2, -1, -1):
        res_x[i] = u[i] * res_x[i + 1] + v[i]
    return res_x


def spline_interp(x, y, xq):
    n = len(x)
    a = [0] * n
    b = [0] * n
    c = [0] * n
    d = [0] * n
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    for i in range(1, n - 1):
        a[i] = h[i - 1]
        b[i] = 2 * (h[i - 1] + h[i])
        c[i] = h[i]
        d[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    b[0] = b[-1] = 1
    d[0] = d[-1] = 0
    c[0] = a[-1] = 0
    m_coeffs = run_through(a, b, c, d)
    for i in range(1, n):
        if xq <= x[i]:
            hi = x[i] - x[i - 1]
            alpha = (x[i] - xq) / hi
            beta = (xq - x[i - 1]) / hi
            s = (alpha * y[i - 1] + beta * y[i] +
                 ((alpha ** 3 - alpha) * m_coeffs[i - 1] + (beta ** 3 - beta) * m_coeffs[i]) *
                 (hi ** 2) / 6)
            return s
    return y[-1]


def newton_interp(x, y, xq):
    n = len(x)
    m = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        m[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            m[i][j] = (m[i + 1][j - 1] - m[i][j - 1]) / (x[i + j] - x[i])
    s = m[0][0]
    for j in range(1, n):
        p = 1
        for k in range(j):
            p *= (xq - x[k])
        s += m[0][j] * p
    return s


def mnk_quadratic(x, y, xq):
    n = len(x)
    sx = sum(x)
    sx2 = sum(xi ** 2 for xi in x)
    sx3 = sum(xi ** 3 for xi in x)
    sx4 = sum(xi ** 4 for xi in x)
    sy = sum(y)
    sxy = sum(x[i] * y[i] for i in range(n))
    sx2y = sum((x[i] ** 2) * y[i] for i in range(n))
    mat_a = [[n, sx, sx2],
             [sx, sx2, sx3],
             [sx2, sx3, sx4]]
    vec_b = [sy, sxy, sx2y]

    def gauss(a_in, b_in):
        n_size = len(b_in)
        for i in range(n_size):
            max_row = i
            for j in range(i + 1, n_size):
                if abs(a_in[j][i]) > abs(a_in[max_row][i]):
                    max_row = j
            a_in[i], a_in[max_row] = a_in[max_row], a_in[i]
            b_in[i], b_in[max_row] = b_in[max_row], b_in[i]
            for j in range(i + 1, n_size):
                ratio = a_in[j][i] / a_in[i][i]
                for k in range(i, n_size):
                    a_in[j][k] -= ratio * a_in[i][k]
                b_in[j] -= ratio * b_in[i]
        res_x = [0 for _ in range(n_size)]
        for i in range(n_size - 1, -1, -1):
            res_x[i] = (b_in[i] - sum(a_in[i][j] * res_x[j] for j in range(i + 1, n_size))) / \
                a_in[i][i]
        return res_x

    c0, c1, c2 = gauss(mat_a, vec_b)
    return c0 + c1 * xq + c2 * (xq ** 2), (c0, c1, c2)


def print_table(title, func):
    print(f"\n{title}")
    print("=" * len(title))
    print(f"{'x':>6} {'y(x)':>12}")
    for xi in np.arange(0.1, 2.01, 0.1):
        yi = func(X_DATA, Y_DATA, xi)
        print(f"{xi:6.2f} {yi:12.5f}")
    print("\n--- Интерполяция между точками ---")
    for xi in np.arange(0.15, 1.96, 0.2):
        yi = func(X_DATA, Y_DATA, xi)
        print(f"{xi:6.2f} {yi:12.5f}")


def spline3_fixed(x, y):
    n = len(x)
    h = np.diff(x)
    alpha = np.zeros(n)

    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - \
            (3 / h[i - 1]) * (y[i] - y[i - 1])

    l_arr = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l_arr[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l_arr[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l_arr[i]

    l_arr[-1] = 1
    z[-1] = 0

    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    a = y[:-1]
    return a, b, c[:-1], d


def dif_spline_fixed(x, coeffs, xx):
    a, b, c, d = coeffs
    i = np.searchsorted(x, xx) - 1
    if i < 0:
        i = 0
    if i >= len(a):
        i = len(a) - 1

    dx = xx - x[i]

    f1 = b[i] + 2 * c[i] * dx + 3 * d[i] * dx**2
    f2 = 2 * c[i] + 6 * d[i] * dx
    return f1, f2


print_table("Интерполяция полиномом Лагранжа", lagrange_interp)
print_table("Интерполяция кубическим сплайном", spline_interp)
print_table("Интерполяция полиномом Ньютона", newton_interp)
print_table("Аппроксимация методом наименьших квадратов",
            lambda x, y, xi: mnk_quadratic(x, y, xi)[0])

_, (coeff_0, coeff_1, coeff_2) = mnk_quadratic(X_DATA, Y_DATA, 0)
print(f"\nКоэффициенты МНК: c0 = {coeff_0:.6f}, c1 = {coeff_1:.6f}, c2 = {coeff_2:.6f}")


xq_vals = np.linspace(0.1, 2.0, 300)

lag_sci = lagrange(X_DATA, Y_DATA)
spl_sci = CubicSpline(X_DATA, Y_DATA)
coef_mnk = np.polyfit(X_DATA, Y_DATA, 2)

methods = {
    "Полином Лагранжа": [lambda xi: lagrange_interp(X_DATA, Y_DATA, xi),
                         lambda xi: lag_sci(xi), 'r'],
    "Кубический сплайн": [lambda xi: spline_interp(X_DATA, Y_DATA, xi),
                          lambda xi: spl_sci(xi), 'b'],
    "Полином Ньютона": [lambda xi: newton_interp(X_DATA, Y_DATA, xi),
                        None, 'g'],
    "МНК (квадратичная аппроксимация)": [
        lambda xi: mnk_quadratic(X_DATA, Y_DATA, xi)[0],
        lambda xi: np.polyval(coef_mnk, xi), 'm'
    ]
}

for name, (manual, builtin, color) in methods.items():
    y_manual = [manual(xi) for xi in xq_vals]
    y_builtin = [builtin(xi) for xi in xq_vals] if builtin else None

    plt.figure(figsize=(10, 6))
    plt.scatter(X_DATA, Y_DATA, color='black', label='Исходные точки', zorder=5, s=40)
    plt.plot(xq_vals, y_manual, color=color, linewidth=2, label=f'{name} (ручной)')
    if builtin:
        plt.plot(xq_vals, y_builtin, color=color, linestyle='--',
                 linewidth=1.5, label=f'{name} (встроенный)')
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    def func_f(x):
        return math.cos(x + x**3)

    def func_fx(x):
        return -math.sin(x + x**3) * (1 + 3 * x**2)

    def func_fxx(x):
        return (
            -math.cos(x + x**3) * (1 + 3 * x**2)**2
            - math.sin(x + x**3) * (6 * x)
        )

    val_a, val_b, val_m = 0, 1, 15
    x_nodes = np.linspace(val_a, val_b, val_m)
    y_nodes = np.array([func_f(xi) for xi in x_nodes])

    print(" i      x_i             y_i")
    for idx, (xi, yi) in enumerate(zip(x_nodes, y_nodes)):
        print(f"{idx:2d} | {xi:10.6f} | {yi:12.6f}")
    print()

    spline_coeffs = spline3_fixed(x_nodes, y_nodes)
    target_x = 0.55
    res_s1, res_s2 = dif_spline_fixed(x_nodes, spline_coeffs, target_x)

    cs_sci = CubicSpline(x_nodes, y_nodes, bc_type='natural')
    s1_builtin = cs_sci(target_x, 1)
    s2_builtin = cs_sci(target_x, 2)

    fx_real = func_fx(target_x)
    fxx_real = func_fxx(target_x)

    print(f"x = {target_x}")
    print("------ Сравнение ------")
    print(f"Наша реализация:     f'(x) = {res_s1:.6f}, f''(x) = {res_s2:.6f}")
    print(f"SciPy CubicSpline:   f'(x) = {s1_builtin:.6f}, "
          f"f''(x) = {s2_builtin:.6f}")
    print(f"Аналитически:        f'(x) = {fx_real:.6f}, "
          f"f''(x) = {fxx_real:.6f}")

    print("\nПогрешности:")
    print(f"Δf'(x): {abs(res_s1 - fx_real):.6e}")
    print(f"Δf''(x): {abs(res_s2 - fxx_real):.6e}")

    xs_plot = np.linspace(val_a, val_b, 300)
    ys_plot = [func_f(xi) for xi in xs_plot]
    dydx_plot = [func_fx(xi) for xi in xs_plot]
    d2ydx2_plot = [func_fxx(xi) for xi in xs_plot]

    ys_spline = cs_sci(xs_plot)
    dydx_spline = cs_sci(xs_plot, 1)
    d2ydx2_spline = cs_sci(xs_plot, 2)

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.title("Численное дифференцирование: f(x) = cos(x + x³)")
    plt.plot(xs_plot, ys_plot, label="f(x)")
    plt.plot(xs_plot, ys_spline, '--', label="Сплайн SciPy")
    plt.scatter(x_nodes, y_nodes, color='black', s=20)
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(xs_plot, dydx_plot, label="f'(x) аналитически")
    plt.plot(xs_plot, dydx_spline, '--', label="f'(x) сплайн")
    plt.axvline(target_x, color='gray', linestyle='--')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(xs_plot, d2ydx2_plot, label="f''(x) аналитически")
    plt.plot(xs_plot, d2ydx2_spline, '--', label="f''(x) сплайн")
    plt.axvline(target_x, color='gray', linestyle='--')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
