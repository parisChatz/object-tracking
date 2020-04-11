x_pred = []
vx_pred = []
y_pred = []
vy_pred = []

Zx = []
Zy = []

Px = []
Pvx = []
Py = []
Pvy = []

Rvx = []
Rvy = []

Kx = []
Kvx = []
Ky = []
Kvy = []


def append_states(x, z, p, r, k):
    x_pred.append(float(x[0]))
    vx_pred.append(float(x[1]))
    y_pred.append(float(x[2]))
    vy_pred.append(float(x[3]))

    Zx.append(float(z[0]))
    Zy.append(float(z[1]))

    Px.append(float(p[0, 0]))
    Pvx.append(float(p[1, 1]))
    Py.append(float(p[2, 2]))
    Pvy.append(float(p[3, 3]))

    Rvx.append(float(r[0, 0]))
    Rvy.append(float(r[1, 1]))

    Kx.append(float(k[0, 0]))
    Kvx.append(float(k[1, 0]))
    Ky.append(float(k[2, 0]))
    Kvy.append(float(k[3, 0]))
