import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


x = [i * 0.1 for i in range(1, 21)]
y = [0.17, 0.07, 0.17, 0.05, 0.12, 0.00, 0.01, -0.05, -0.21, -0.50,
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


def progonka(a, b, c, d):
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
    x = [0] * n
    x[-1] = v[-1]
    for i in range(n - 2, -1, -1):
        x[i] = u[i] * x[i + 1] + v[i]
    return x


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
    M = progonka(a, b, c, d)
    for i in range(1, n):
        if xq <= x[i]:
            hi = x[i] - x[i - 1]
            A = (x[i] - xq) / hi
            B = (xq - x[i - 1]) / hi
            s = A * y[i - 1] + B * y[i] + ((A ** 3 - A) * M[i - 1] + (B ** 3 - B) * M[i]) * (hi ** 2) / 6
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
    A = [[n, sx, sx2],
         [sx, sx2, sx3],
         [sx2, sx3, sx4]]
    B = [sy, sxy, sx2y]

    def gauss(A, B):
        n = len(B)
        for i in range(n):
            max_row = i
            for j in range(i + 1, n):
                if abs(A[j][i]) > abs(A[max_row][i]):
                    max_row = j
            A[i], A[max_row] = A[max_row], A[i]
            B[i], B[max_row] = B[max_row], B[i]
            for j in range(i + 1, n):
                ratio = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] -= ratio * A[i][k]
                B[j] -= ratio * B[i]
        x = [0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            x[i] = (B[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
        return x

    c0, c1, c2 = gauss(A, B)
    return c0 + c1 * xq + c2 * (xq ** 2), (c0, c1, c2)


def print_table(title, func):
    print(f"\n{title}")
    print("=" * len(title))
    print(f"{'x':>6} {'y(x)':>12}")
    for xi in np.arange(0.1, 2.01, 0.1):
        yi = func(x, y, xi)
        print(f"{xi:6.2f} {yi:12.5f}")
    print("\n--- Интерполяция между точками ---")
    for xi in np.arange(0.15, 1.96, 0.2):
        yi = func(x, y, xi)
        print(f"{xi:6.2f} {yi:12.5f}")


def spline3_fixed(x, y):
    n = len(x)
    h = np.diff(x)
    alpha = np.zeros(n)

    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[-1] = 1
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
print_table("Аппроксимация методом наименьших квадратов", lambda x, y, xi: mnk_quadratic(x, y, xi)[0])

_, (c0, c1, c2) = mnk_quadratic(x, y, 0)
print(f"\nКоэффициенты МНК: c0 = {c0:.6f}, c1 = {c1:.6f}, c2 = {c2:.6f}")


xq = np.linspace(0.1, 2.0, 300)

lag_sci = lagrange(x, y)
spl_sci = CubicSpline(x, y)
coef_mnk = np.polyfit(x, y, 2)

methods = {
    "Полином Лагранжа": [lambda xi: lagrange_interp(x, y, xi),
                         lambda xi: lag_sci(xi), 'r'],
    "Кубический сплайн": [lambda xi: spline_interp(x, y, xi),
                          lambda xi: spl_sci(xi), 'b'],
    "Полином Ньютона": [lambda xi: newton_interp(x, y, xi),
                        None, 'g'],
    "МНК (квадратичная аппроксимация)": [lambda xi: mnk_quadratic(x, y, xi)[0],
                                         lambda xi: np.polyval(coef_mnk, xi), 'm']
}

for name, (manual, builtin, color) in methods.items():
    y_manual = [manual(xi) for xi in xq]
    y_builtin = [builtin(xi) for xi in xq] if builtin else None

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='black', label='Исходные точки', zorder=5, s=40)
    plt.plot(xq, y_manual, color=color, linewidth=2, label=f'{name} (ручной)')
    if builtin:
        plt.plot(xq, y_builtin, color=color, linestyle='--', linewidth=1.5, label=f'{name} (встроенный)')
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    def f(x):
        return math.cos(x + x**3)

    def fx(x):
        return -math.sin(x + x**3) * (1 + 3 * x**2)

    def fxx(x):
        return (
            -math.cos(x + x**3) * (1 + 3 * x**2)**2
            - math.sin(x + x**3) * (6 * x)
        )

    a, b, m = 0, 1, 15
    x = np.linspace(a, b, m)
    y = np.array([f(xi) for xi in x])

    print(" i      x_i             y_i")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"{i:2d} | {xi:10.6f} | {yi:12.6f}")
    print()

    coeffs = spline3_fixed(x, y)
    xx = 0.55
    s1, s2 = dif_spline_fixed(x, coeffs, xx)

    cs = CubicSpline(x, y, bc_type='natural')
    s1_builtin = cs(xx, 1)
    s2_builtin = cs(xx, 2)

    fx_real = fx(xx)
    fxx_real = fxx(xx)

    print(f"x = {xx}")
    print("------ Сравнение ------")
    print(f"Наша реализация:     f'(x) = {s1:.6f}, f''(x) = {s2:.6f}")
    print(f"SciPy CubicSpline:   f'(x) = {s1_builtin:.6f}, f''(x) = {s2_builtin:.6f}")
    print(f"Аналитически:        f'(x) = {fx_real:.6f}, f''(x) = {fxx_real:.6f}")

    print("\nПогрешности:")
    print(f"Δf'(x): {abs(s1 - fx_real):.6e}")
    print(f"Δf''(x): {abs(s2 - fxx_real):.6e}")

    xs = np.linspace(a, b, 300)
    ys = [f(xi) for xi in xs]
    dydx = [fx(xi) for xi in xs]
    d2ydx2 = [fxx(xi) for xi in xs]

    ys_spline = cs(xs)
    dydx_spline = cs(xs, 1)
    d2ydx2_spline = cs(xs, 2)

    plt.figure(figsize=(10, 7))

    plt.subplot(3, 1, 1)
    plt.title("Численное дифференцирование: f(x) = cos(x + x³)")
    plt.plot(xs, ys, label="f(x)")
    plt.plot(xs, ys_spline, '--', label="Сплайн SciPy")
    plt.scatter(x, y, color='black', s=20)
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(xs, dydx, label="f'(x) аналитически")
    plt.plot(xs, dydx_spline, '--', label="f'(x) сплайн")
    plt.axvline(xx, color='gray', linestyle='--')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(xs, d2ydx2, label="f''(x) аналитически")
    plt.plot(xs, d2ydx2_spline, '--', label="f''(x) сплайн")
    plt.axvline(xx, color='gray', linestyle='--')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()