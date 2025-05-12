from itertools import chain

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
from matplotlib import cm
from numba import njit

FREQ = 40e3  # Frequency of ultrasound
C = 340e3  # Speed of sound
WAVELENGTH = C / FREQ
WAVENUMBER = 2 * np.pi / WAVELENGTH

TRANS_SPACING = 10.16
N_TRANS_IN_X = 18
N_TRANS_IN_Y = 14

FOCUS_Z = 150

# field plot size
RESOLUTION = 2
Nx = 41
Ny = 41

# optimization parameter
PHASE_DIV = 16


def adjust_marker_size(fig, axes):
    radius = 10.0 / 2
    fig.canvas.draw()
    r_pix = axes.transData.transform((radius, radius)) - axes.transData.transform(
        (0, 0)
    )
    sizes = (2 * r_pix * 72 / fig.dpi) ** 2
    return sizes[0]


def add_colorbar(fig, axes, mappable):
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(axes)
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cm.jet), cax=cax)
    cbar.ax.set_ylim((0, 1))
    cbar.ax.set_yticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])


def plot_trans(axes, trans_x, trans_y, trans_amp, trans_phase, marker_size):
    scat = axes.scatter(
        trans_x,
        trans_y,
        c=[
            (*cm.jet(trans_phase[k] / (2 * np.pi))[:3], trans_amp[k])
            for k in range(len(trans_phase))
        ],
        s=marker_size,
        marker="o",
        clip_on=False,
    )
    return scat


def plot_field(axes, x, y, p):
    TICK_STEP = 10

    heatmap = axes.pcolor(np.abs(p).T, cmap="jet")
    x_label_num = int(np.floor((x[-1] - x[0]) / TICK_STEP)) + 1
    y_label_num = int(np.floor((y[-1] - y[0]) / TICK_STEP)) + 1
    x_labels = ["{:.2f}".format(x[0] + TICK_STEP * i) for i in range(x_label_num)]
    y_labels = ["{:.2f}".format(y[0] + TICK_STEP * i) for i in range(y_label_num)]
    x_ticks = [TICK_STEP / RESOLUTION * i for i in range(x_label_num)]
    y_ticks = [TICK_STEP / RESOLUTION * i for i in range(y_label_num)]
    axes.set_xticks(np.array(x_ticks) + 0.5, minor=False)
    axes.set_yticks(np.array(y_ticks) + 0.5, minor=False)
    axes.set_xticklabels(x_labels, minor=False)
    axes.set_yticklabels(y_labels, minor=False)

    return heatmap


@njit
def propagate(source, target) -> np.complex128:
    return np.exp(-1j * WAVENUMBER * np.linalg.norm(target - source)) / np.linalg.norm(
        target - source
    )


@njit(parallel=True)
def calc_field(trans_x, trans_y, trans_amp, trans_phase, field_x, field_y):
    f = np.zeros((len(field_x), len(field_y)), dtype=np.complex128)

    for x in range(len(field_x)):
        for y in range(len(field_y)):
            target = np.array([field_x[x], field_y[y], FOCUS_Z])
            p = 0 + 0j
            for i in range(len(trans_x)):
                ta = trans_amp[i] + 0j
                p += (
                    ta
                    * propagate(np.array([trans_x[i], trans_y[i], 0]), target)
                    * np.exp(1j * trans_phase[i])
                )

            f[x, y] = p

    return f


def single():
    fig = plt.figure(figsize=(10, 12), dpi=72)
    ax_field = fig.add_subplot(2, 1, 1, aspect="equal")
    ax_trans = fig.add_subplot(2, 1, 2, aspect="equal")

    trans_x = np.fromiter(
        chain.from_iterable(
            [
                [TRANS_SPACING * x for x in range(N_TRANS_IN_X)]
                for _ in range(N_TRANS_IN_Y)
            ]
        ),
        dtype=float,
    )
    trans_y = np.fromiter(
        chain.from_iterable(
            [
                [TRANS_SPACING * y for _ in range(N_TRANS_IN_X)]
                for y in range(N_TRANS_IN_Y)
            ]
        ),
        dtype=float,
    )
    # translate array to set origin at the center
    trans_x -= np.mean(trans_x)
    trans_y -= np.mean(trans_y)

    trans_amp = np.zeros(N_TRANS_IN_X * N_TRANS_IN_Y, dtype=float)
    trans_phase = np.zeros(N_TRANS_IN_X * N_TRANS_IN_Y, dtype=float)

    field_x = np.linspace(-(Nx - 1) // 2 * RESOLUTION, (Nx - 1) // 2 * RESOLUTION, Nx)
    field_y = np.linspace(-(Ny - 1) // 2 * RESOLUTION, (Ny - 1) // 2 * RESOLUTION, Ny)

    indices = np.arange(N_TRANS_IN_X * N_TRANS_IN_Y)
    np.random.shuffle(indices)

    # set marker size as the same size as the transducer
    plt.tight_layout()
    plot_field(
        ax_field,
        field_x,
        field_y,
        calc_field(trans_x, trans_y, trans_amp, trans_phase, field_x, field_y),
    )
    scat = plot_trans(ax_trans, trans_x, trans_y, trans_amp, trans_phase, 0)
    add_colorbar(fig, ax_trans, scat)
    marker_size = adjust_marker_size(fig, ax_trans)

    global cache
    cache = 0 + 0j
    phase_options = np.linspace(0, 2 * np.pi, PHASE_DIV, endpoint=False)
    target_amp = 10  # sufficiently large value

    FOCUS_X = 0
    FOCUS_Y = 0

    def plot(n):
        global cache

        if n != 0:
            ax_trans.cla()
            ax_field.cla()

        i = indices[n]

        pp = propagate(
            np.array([trans_x[i], trans_y[i], 0]), np.array([FOCUS_X, FOCUS_Y, FOCUS_Z])
        )
        err_min = np.inf
        phase_min = 0
        for phase in phase_options:
            err = np.abs(target_amp - np.abs(cache + pp * np.exp(1j * phase)))
            if err < err_min:
                err_min = err
                phase_min = phase

        cache += pp * np.exp(1j * phase_min)

        trans_phase[i] = phase_min
        trans_amp[i] = 1.0

        p = calc_field(trans_x, trans_y, trans_amp, trans_phase, field_x, field_y)
        plot_field(ax_field, field_x, field_y, p)
        plot_trans(ax_trans, trans_x, trans_y, trans_amp, trans_phase, marker_size)

    ani = animation.FuncAnimation(
        fig, plot, frames=len(indices), interval=10, repeat=False
    )

    plt.show()


def multi():
    fig = plt.figure(figsize=(10, 12), dpi=72)
    ax_field = fig.add_subplot(2, 1, 1, aspect="equal")
    ax_trans = fig.add_subplot(2, 1, 2, aspect="equal")

    trans_x = np.fromiter(
        chain.from_iterable(
            [
                [TRANS_SPACING * x for x in range(N_TRANS_IN_X)]
                for _ in range(N_TRANS_IN_Y)
            ]
        ),
        dtype=float,
    )
    trans_y = np.fromiter(
        chain.from_iterable(
            [
                [TRANS_SPACING * y for _ in range(N_TRANS_IN_X)]
                for y in range(N_TRANS_IN_Y)
            ]
        ),
        dtype=float,
    )
    # translate array to set origin at the center
    trans_x -= np.mean(trans_x)
    trans_y -= np.mean(trans_y)

    trans_amp = np.zeros(N_TRANS_IN_X * N_TRANS_IN_Y, dtype=float)
    trans_phase = np.zeros(N_TRANS_IN_X * N_TRANS_IN_Y, dtype=float)

    field_x = np.linspace(-(Nx - 1) // 2 * RESOLUTION, (Nx - 1) // 2 * RESOLUTION, Nx)
    field_y = np.linspace(-(Ny - 1) // 2 * RESOLUTION, (Ny - 1) // 2 * RESOLUTION, Ny)

    indices = np.arange(N_TRANS_IN_X * N_TRANS_IN_Y)
    np.random.shuffle(indices)

    # set marker size as the same size as the transducer
    plt.tight_layout()
    plot_field(
        ax_field,
        field_x,
        field_y,
        calc_field(trans_x, trans_y, trans_amp, trans_phase, field_x, field_y),
    )
    scat = plot_trans(ax_trans, trans_x, trans_y, trans_amp, trans_phase, 0)
    add_colorbar(fig, ax_trans, scat)
    marker_size = adjust_marker_size(fig, ax_trans)

    foci = np.array(
        [
            [-20, 0, FOCUS_Z],
            [20, 0, FOCUS_Z],
        ]
    )

    global cache
    cache = np.zeros(len(foci), dtype=np.complex128)
    phase_options = np.linspace(0, 2 * np.pi, PHASE_DIV, endpoint=False)
    target_amp = (
        np.ones(len(foci), dtype=np.complex128) * 10
    )  # sufficiently large value

    def plot(n):
        global cache

        if n != 0:
            ax_trans.cla()
            ax_field.cla()

        i = indices[n]

        pp = np.fromiter(
            (propagate(np.array([trans_x[i], trans_y[i], 0]), f) for f in foci),
            dtype=np.complex128,
        )
        err_min = np.inf
        phase_min = 0
        for phase in phase_options:
            err = np.abs(target_amp - np.abs(cache + pp * np.exp(1j * phase))).sum()
            if err < err_min:
                err_min = err
                phase_min = phase

        cache += pp * np.exp(1j * phase_min)

        trans_phase[i] = phase_min
        trans_amp[i] = 1.0

        p = calc_field(trans_x, trans_y, trans_amp, trans_phase, field_x, field_y)
        plot_field(ax_field, field_x, field_y, p)
        plot_trans(ax_trans, trans_x, trans_y, trans_amp, trans_phase, marker_size)

    ani = animation.FuncAnimation(
        fig, plot, frames=len(indices), interval=10, repeat=False
    )

    plt.show()


if __name__ == "__main__":
    single()
    multi()
