"""
Visualizing SGD behavior (momentum, weight_decay, maximize).
Optimizes a toy 2D quadratic with torch.optim.SGD and plots the (x, y)
trajectory over a contour of the objective. 
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


CURVE_Y = 10.0  # ill-conditioned quadratic: steep in y, shallow in x


def f_min(x, y):
    return x ** 2 + CURVE_Y * y ** 2


def f_max(x, y):
    return -(x ** 2) - CURVE_Y * y ** 2


def run_sgd(fn, start, steps, lr, momentum=0.0, weight_decay=0.0,
            dampening=0.0, nesterov=False, maximize=False, device="cpu"):
    xy = torch.tensor(start, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.SGD(
        [xy], lr=lr, momentum=momentum, weight_decay=weight_decay,
        dampening=dampening, nesterov=nesterov, maximize=maximize,
    )
    traj = [xy.detach().cpu().numpy().copy()]
    for _ in range(steps):
        opt.zero_grad()
        loss = fn(xy[0], xy[1])
        loss.backward()
        opt.step()
        traj.append(xy.detach().cpu().numpy().copy())
    return np.array(traj)


def contour_grid(fn, lim=4.5, n=200):
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, ys)
    Z = fn(X, Y)
    return X, Y, Z


def plot_trajectories(ax, fn, trajectories, labels, title, lim=4.5, linewidths=None):
    X, Y, Z = contour_grid(fn, lim=lim)
    ax.contour(X, Y, Z, levels=20, cmap="Greys", alpha=0.5)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(trajectories)))
    if linewidths is None:
        linewidths = [1.2] * len(trajectories)
    for traj, label, c, lw in zip(trajectories, labels, colors, linewidths):
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=2, lw=lw, color=c, label=label)
        ax.plot(traj[0, 0], traj[0, 1], "s", ms=6, color=c)  # start
        ax.plot(traj[-1, 0], traj[-1, 1], "*", ms=10, color=c)  # end
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--start", type=float, nargs=2, default=[4.0, 1.2])
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")
    momenta = [0.0, 0.5, 0.9]

    # (a) momentum sweep, no weight decay
    trajs_a = [
        run_sgd(f_min, args.start, args.steps, args.lr, momentum=m, device=device)
        for m in momenta
    ]
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_trajectories(
        ax, f_min, trajs_a, [f"momentum={m}" for m in momenta],
        "SGD on f(x,y)=x²+10y² : effect of momentum",
    )
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/momentum_trajectories.png", dpi=150)
    plt.close(fig)

    # (b) same momenta, with weight_decay=0.1
    trajs_b = [
        run_sgd(f_min, args.start, args.steps, args.lr, momentum=m,
                weight_decay=0.1, device=device)
        for m in momenta
    ]
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_trajectories(
        ax, f_min, trajs_b, [f"momentum={m}" for m in momenta],
        "SGD on f(x,y)=x²+10y² : momentum + weight_decay=0.1",
    )
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/weight_decay_trajectories.png", dpi=150)
    plt.close(fig)

    # (c) maximize=True on -x^2-y^2 vs. minimize on x^2+y^2 (same optimum at origin)
    traj_min = run_sgd(f_min, args.start, args.steps, args.lr, momentum=0.9, device=device)
    traj_max = run_sgd(f_max, args.start, args.steps, args.lr, momentum=0.9,
                        maximize=True, device=device)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_trajectories(
        ax, f_max, [traj_min, traj_max],
        ["minimize f=x²+10y² (maximize=False)", "maximize f=-x²-10y² (maximize=True)"],
        "maximize=True flips the gradient sign (trajectories coincide exactly)",
        linewidths=[4.0, 1.2],
    )
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/maximize_trajectories.png", dpi=150)
    plt.close(fig)

    # sanity check: the two trajectories in (c) should coincide (sign flip is correct)
    max_diff = np.abs(traj_min - traj_max).max()
    print(f"(c) max |traj_min - traj_max| over all steps: {max_diff:.2e}  (should be ~0)")

    for name, trajs in [("momentum", trajs_a), ("weight_decay", trajs_b)]:
        for m, t in zip(momenta, trajs):
            print(f"{name} m={m}: final (x,y)={t[-1]}, dist_to_origin={np.linalg.norm(t[-1]):.4f}")

    print("Saved: momentum_trajectories.png, weight_decay_trajectories.png, maximize_trajectories.png")


if __name__ == "__main__":
    main()

