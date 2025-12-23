from acados_template import AcadosOcpSolver
import numpy as np
from typing import Dict, Any
import time

# debugging
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

class StriderNMPC:
    def __init__(self):
        from .model import build_ocp
        self.ocp = build_ocp()

        json_path = BASE_DIR / f"{self.ocp.model.name}.json"
        self.solver = AcadosOcpSolver(self.ocp, json_file=str(json_path))
        self.nx = self.ocp.model.x.size()[0]
        self.nu = self.ocp.model.u.size()[0]
        self.np = self.ocp.model.p.size()[0]
        
        from .params import N, MASS, G_ACCEL
        self.N = int(N)
        self.f_ref = float(MASS*G_ACCEL)

        # debug
        from .params import DT
        self.DT = float(DT)
        self.last_debug_states: list[np.ndarray] | None = None
        self.last_debug_solve_ms: float | None = None
        self.last_debug_status: int | None = None
        self.last_debug_pos_ref: np.ndarray | None = None
        self.solve_ms_hist = np.zeros(50, dtype=np.float64)
        self.solve_ms_count = 0
    
    def solve(self, pos_d, yaw_d, x_0, u_0, p, debug: bool = False):
        x_0   = np.asarray(x_0,   dtype=np.float64).ravel()
        u_0   = np.asarray(u_0,   dtype=np.float64).ravel()
        p     = np.asarray(p,     dtype=np.float64).ravel()
        pos_d = np.asarray(pos_d, dtype=np.float64).ravel()
        yaw_d = np.asarray(yaw_d, dtype=np.float64).ravel()

        # initial state condition (equality constraint)
        self.solver.set(0, "lbx", x_0)
        self.solver.set(0, "ubx", x_0)

        # initial guess
        for k in range(self.N + 1): self.solver.set(k, "x", x_0)
        for k in range(self.N): self.solver.set(k, "u", u_0)

        # feed parameter
        for k in range(self.N + 1): self.solver.set(k, "p", p)

        # feed input reference
        # [0,k-1]th step reference [p_d, theta_d, f_d]
        for k in range(self.N): self.solver.set(k, "yref", np.concatenate([pos_d[3*k : 3*(k+1)], np.array([float(yaw_d[k]), self.f_ref], dtype=np.float64)]))
        # [k]th terminal reference [p_d, theta_d]
        self.solver.set(self.N, "yref", np.concatenate([pos_d[3 * (self.N - 1) : 3 * self.N], np.array([float(yaw_d[self.N - 1])], dtype=np.float64)]))

        # Solve
        t0 = time.perf_counter()
        status = self.solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract first control
        if status == 0: u_opt0 = self.solver.get(0, "u").flatten()
        else: u_opt0 = u_0.copy()

        # time count
        idx = self.solve_ms_count % self.solve_ms_hist.size
        self.solve_ms_hist[idx] = solve_ms
        self.solve_ms_count += 1

        if debug:
            xs_debug: list[np.ndarray] = []
            for k in range(self.N + 1):
                xk = self.solver.get(k, "x").flatten().copy()
                xs_debug.append(xk)

            self.last_debug_states = xs_debug
            self.last_debug_solve_ms = float(solve_ms)
            self.last_debug_status = int(status)
            self.last_debug_pos_ref = pos_d.reshape(self.N, 3).copy()
            self.last_debug_yaw_ref = yaw_d.copy()

        return u_opt0.astype(np.float64), float(solve_ms), int(status)
    
    def compute_MPC(self, mpci: Dict[str, Any]) -> Dict[str, Any]:
        pos_d  = np.asarray(mpci.get("pos_d", np.zeros(3*self.N)), dtype=np.float64).ravel()
        yaw_d  = np.asarray(mpci.get("yaw_d", np.zeros(self.N)),   dtype=np.float64).ravel()
        x_0    = np.asarray(mpci.get("x_0",   np.zeros(self.nx)),  dtype=np.float64).ravel()
        u_0    = np.asarray(mpci.get("u_0",   np.zeros(self.nu)),  dtype=np.float64).ravel()
        p      = np.asarray(mpci.get("p",     np.zeros(self.np)),  dtype=np.float64).ravel()
        debug  = bool(mpci.get("debug", False))
        
        if x_0.size != self.nx: raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.nx}")
        if p.size != self.np:   raise ValueError(f"p size mismatch: got {p.size}, expected {self.np}")
        if u_0.size != self.nu: u_0 = np.zeros(self.nu, dtype=np.float64)

        q_d, solve_ms, status = self.solve(pos_d, yaw_d, x_0, u_0, p, debug=debug)

        return {"u": q_d.astype(np.float64), "solve_ms": float(solve_ms), "state": int(status),}

    def print_last_debug(self) -> None: # plot func
        # Do Not use at realtime(only Debug)
        from .params import DH_PARAMS_ARM, DH_PARAMS_BASE, MASS, G_ACCEL, COST_POS_ERR, COST_ANG_ERR, COST_F_THRUST
        states = np.stack(self.last_debug_states, axis=0)
        T, _ = states.shape
        p  = states[:, 0:3]    # p_cot (global CoT position)
        v  = states[:, 3:6]    # v_cot
        th = states[:, 6:9]    # [roll, pitch, yaw]
        om = states[:, 9:12]   # omega
        q  = states[:, 12:32]  # 20 joint angles
        p_param = np.array(self.solver.get(0, "p")).ravel()
        dt = float(getattr(self, "DT", 1.0))
        t  = np.arange(T, dtype=np.float64) * dt
        cot_p_c_hat = p_param[0:3]
        g_joints_hist = np.zeros((T, 4, 6, 3), dtype=np.float64)

        def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
            cr = np.cos(roll); sr = np.sin(roll)
            cp = np.cos(pitch); sp = np.sin(pitch)
            cy = np.cos(yaw);   sy = np.sin(yaw)
            Rz = np.array([[cy, -sy, 0.0], [sy,  cy, 0.0], [0.0, 0.0, 1.0],])
            Ry = np.array([[ cp, 0.0,  sp], [0.0, 1.0, 0.0], [-sp, 0.0,  cp],])
            Rx = np.array([[1.0, 0.0, 0.0], [0.0,  cr, -sr], [0.0,  sr,  cr],])
            return Rz @ Ry @ Rx
        
        def dh_transform_np(a_link, alpha_joint, d_link, theta_joint):
            ct, st = np.cos(theta_joint), np.sin(theta_joint)
            ca_, sa_ = np.cos(alpha_joint), np.sin(alpha_joint)
            Tm = np.zeros((4, 4), dtype=np.float64)
            Tm[0, 0] = ct
            Tm[0, 1] = -st * ca_
            Tm[0, 2] = st * sa_
            Tm[0, 3] = a_link * ct
            Tm[1, 0] = st
            Tm[1, 1] = ct * ca_
            Tm[1, 2] = -ct * sa_
            Tm[1, 3] = a_link * st
            Tm[2, 0] = 0.0
            Tm[2, 1] = sa_
            Tm[2, 2] = ca_
            Tm[2, 3] = d_link
            Tm[3, 3] = 1.0
            return Tm

        def FK_chain_np(q_arm, arm_idx):
            # body frame: B -> each joint (0..5), including tip at index 5
            a0     = dh_base[arm_idx, 0]
            theta0 = dh_base[arm_idx, 1]
            Tm = dh_transform_np(a0, 0.0, 0.0, theta0)
            joint_pos = np.zeros((6, 3), dtype=np.float64)
            joint_pos[0, :] = Tm[0:3, 3]
            for i in range(dh_arm.shape[0]):
                a     = dh_arm[i, 0]
                alpha = dh_arm[i, 1]
                Tm = Tm @ dh_transform_np(a, alpha, 0.0, q_arm[i])
                joint_pos[i + 1, :] = Tm[0:3, 3]
            return joint_pos

        def update_slider(val):
            # map time value to index
            idx = int(round(val / dt))
            if idx < 0:
                idx = 0
            if idx >= T:
                idx = T - 1

            for arm_idx in range(4):
                gj = g_joints_hist[idx, arm_idx, :, :]
                joint_lines[arm_idx].set_data(gj[:, 0], gj[:, 1])
                joint_lines[arm_idx].set_3d_properties(gj[:, 2])

            fig.canvas.draw_idle()
        
        fig = plt.figure(figsize=(25, 13))
        gs = fig.add_gridspec(
            4, 4,
            width_ratios=[1.8, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0, 1.0, 1.0],
        )

        # ------------------------------------------------------------------
        # 1st col
        # ------------------------------------------------------------------
        #1 cot pos des<->ref
        ax_traj      = fig.add_subplot(gs[0:2, 0], projection='3d')
        ax_arm_traj  = fig.add_subplot(gs[2:4, 0], projection='3d')

        # p_cot reference trajectory
        pref = self.last_debug_pos_ref
        ax_traj.plot(pref[:, 0], pref[:, 1], pref[:, 2],
            linestyle='--', label="p_cot_ref", color='blue'
        )

        # p_cot actual trajectory
        ax_traj.scatter(p[:, 0], p[:, 1], p[:, 2],
            s=4, label="p_cot", color='red', alpha=0.6
        )

        # cot x-axis
        for k in range(T):
            roll_k, pitch_k, yaw_k = th[k, 0], th[k, 1], th[k, 2]
            Rk = rpy_to_R(roll_k, pitch_k, yaw_k)
            x_axis = Rk @ np.array([1.0, 0.0, 0.0])

            px, py, pz = p[k, :]
            ax_traj.quiver(px, py, pz,
                x_axis[0], x_axis[1], x_axis[2],
                length=0.06, normalize=True, color='red', alpha=0.5
            )

        ax_traj.set_xlabel("x [m]")
        ax_traj.set_ylabel("y [m]")
        ax_traj.set_zlabel("z [m]")
        ax_traj.set_title("cot&arm trajectory")
        ax_traj.legend(loc="upper right")

        # ------------------------------------------------------------------
        # 2nd col
        # ------------------------------------------------------------------

        #1 v_cot x,y,z
        ax_v = fig.add_subplot(gs[0, 1])
        ax_v.plot(t, v[:, 0], label="v_x", color='red')
        ax_v.plot(t, v[:, 1], label="v_y", color='green')
        ax_v.plot(t, v[:, 2], label="v_z", color='blue')
        ax_v.set_ylabel("[m/s]")
        ax_v.set_title("v_cot")
        ax_v.legend(loc="upper right")
        ax_v.grid(True)

        #2 roll, pitch, yaw
        ax_th = fig.add_subplot(gs[1, 1])
        roll = th[:, 0] * 180.0 / np.pi
        pitch = th[:, 1] * 180.0 / np.pi
        yaw = th[:, 2] * 180.0 / np.pi
        ax_th.plot(t, roll, label="roll", color='red')
        ax_th.plot(t, pitch, label="pitch", color='green')
        ax_th.plot(t, yaw, label="yaw", color='blue')
        ax_th.set_ylabel("[deg]")
        ax_th.set_title("RPY")
        ax_th.legend(loc="upper right")
        ax_th.grid(True)

        #3 omega
        ax_om = fig.add_subplot(gs[2, 1])
        ax_om.plot(t, om[:, 0], label="Omega_x", color='red')
        ax_om.plot(t, om[:, 1], label="Omega_y", color='green')
        ax_om.plot(t, om[:, 2], label="Omega_z", color='blue')
        ax_om.set_ylabel("[rad/s]")
        ax_om.set_title("Angular velocity")
        ax_om.legend(loc="upper right")
        ax_om.grid(True)

        #4 f_thrust
        ax_f = fig.add_subplot(gs[3, 1])
        N_horizon = self.N
        t_stage = np.arange(N_horizon, dtype=np.float64) * dt
        f_hist = np.zeros(N_horizon, dtype=np.float64)
        for k in range(N_horizon):
            uk = np.array(self.solver.get(k, "u")).ravel()
            f_hist[k] = uk[0]
        ax_f.plot(t_stage, f_hist)
        ax_f.set_xlabel("time [s]")
        ax_f.set_ylabel("[N]")
        ax_f.set_ylim(MASS*G_ACCEL-1.0, MASS*G_ACCEL+1.0)
        ax_f.set_title("total thrust")
        ax_f.grid(True)

        # ------------------------------------------------------------------
        # 3rd col
        # ------------------------------------------------------------------

        dh_arm = np.array(DH_PARAMS_ARM, dtype=np.float64)
        dh_base = np.array(DH_PARAMS_BASE, dtype=np.float64)
        e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        m  = float(MASS)
        g  = float(G_ACCEL)

        cot_pc_hist = np.zeros((T, 3), dtype=np.float64)   # cot -> com (body frame)
        g_p15_hist  = np.zeros((T, 3), dtype=np.float64)   # global -> arm1 tip
        g_p25_hist  = np.zeros((T, 3), dtype=np.float64)   # global -> arm2 tip
        g_p35_hist  = np.zeros((T, 3), dtype=np.float64)   # global -> arm3 tip
        g_p45_hist  = np.zeros((T, 3), dtype=np.float64)   # global -> arm4 tip
        wrench_hist = np.zeros((T, 3), dtype=np.float64)   # wrench at cot

        for k in range(T):
            # FK each arm
            qk = q[k, :]
            q1 = qk[0:5]
            q2 = qk[5:10]
            q3 = qk[10:15]
            q4 = qk[15:20]
            b_joints_1 = FK_chain_np(q1, 0)
            b_joints_2 = FK_chain_np(q2, 1)
            b_joints_3 = FK_chain_np(q3, 2)
            b_joints_4 = FK_chain_np(q4, 3)
            b_p_15 = b_joints_1[-1, :]
            b_p_25 = b_joints_2[-1, :]
            b_p_35 = b_joints_3[-1, :]
            b_p_45 = b_joints_4[-1, :]

            # get cot_p_c
            b_p_cot = (b_p_15 + b_p_25 + b_p_35 + b_p_45) / 4.0
            cot_p_c = cot_p_c_hat - b_p_cot
            cot_pc_hist[k, :] = cot_p_c

            # global -> arm tip = global->cot + R * (cot->arm)
            Rk = rpy_to_R(th[k, 0], th[k, 1], th[k, 2])
            g_p15_hist[k, :] = p[k, :] + Rk @ (b_p_15 - b_p_cot)
            g_p25_hist[k, :] = p[k, :] + Rk @ (b_p_25 - b_p_cot)
            g_p35_hist[k, :] = p[k, :] + Rk @ (b_p_35 - b_p_cot)
            g_p45_hist[k, :] = p[k, :] + Rk @ (b_p_45 - b_p_cot)

            # global joint positions for all joints of each arm
            for arm_idx, b_joints in enumerate([b_joints_1, b_joints_2, b_joints_3, b_joints_4]):
                rel = b_joints - b_p_cot[None, :]
                g_joints = (Rk @ rel.T).T + p[k, :][None, :]
                g_joints_hist[k, arm_idx, :, :] = g_joints

            # wrench
            wrench_hist[k, :] = np.cross(cot_p_c, Rk.T @ (-m * g * e3))

        #1 cot_p_c
        ax_cotpc  = fig.add_subplot(gs[0, 2])
        ax_cotpc.plot(t, 1000.*cot_pc_hist[:, 0], label="cot_p_c_x", color="red")
        ax_cotpc.plot(t, 1000.*cot_pc_hist[:, 1], label="cot_p_c_y", color="green")
        ax_cotpc.plot(t, 1000.*cot_pc_hist[:, 2], label="cot_p_c_z", color="blue")
        ax_cotpc.set_ylabel("[mm]")
        ax_cotpc.set_title("cot_p_com (body frame)")
        ax_cotpc.legend(loc="upper right")
        ax_cotpc.grid(True)

        #2 wrench
        ax_wrench = fig.add_subplot(gs[1, 2])
        ax_wrench.plot(t, wrench_hist[:, 0], label="wrench_x", color="red")
        ax_wrench.plot(t, wrench_hist[:, 1], label="wrench_y", color="green")
        ax_wrench.plot(t, wrench_hist[:, 2], label="wrench_z", color="blue")
        ax_wrench.set_ylabel("[Nm]")
        ax_wrench.set_title("Wrench")
        ax_wrench.legend(loc="upper right")
        ax_wrench.grid(True)

        # 3D plot: arm tip trajectories in global frame
        ax_arm_traj.plot(g_p15_hist[:, 0], g_p15_hist[:, 1], g_p15_hist[:, 2], linestyle='-', marker='.', label="arm1", alpha=0.3, linewidth=1, markersize=2)
        ax_arm_traj.plot(g_p25_hist[:, 0], g_p25_hist[:, 1], g_p25_hist[:, 2], linestyle='-', marker='.', label="arm2", alpha=0.3, linewidth=1, markersize=2)
        ax_arm_traj.plot(g_p35_hist[:, 0], g_p35_hist[:, 1], g_p35_hist[:, 2], linestyle='-', marker='.', label="arm3", alpha=0.3, linewidth=1, markersize=2)
        ax_arm_traj.plot(g_p45_hist[:, 0], g_p45_hist[:, 1], g_p45_hist[:, 2], linestyle='-', marker='.', label="arm4", alpha=0.3, linewidth=1, markersize=2)
        ax_arm_traj.legend(loc="upper right")

        #3 Costs
        pref = self.last_debug_pos_ref    # (N,3)
        yaw_ref = self.last_debug_yaw_ref # (N,)

        N_cost = pref.shape[0]
        t_stage = np.arange(N_cost, dtype=np.float64) * dt

        Wp = np.diag(np.asarray(COST_POS_ERR, dtype=np.float64))
        Wy = float(np.asarray(COST_ANG_ERR, dtype=np.float64).ravel()[0])
        Wf = float(np.asarray(COST_F_THRUST, dtype=np.float64).ravel()[0])

        Jp = np.zeros(N_cost)
        Jy = np.zeros(N_cost)
        Jf = np.zeros(N_cost)
        Jtotal = np.zeros(N_cost + 1)

        for k in range(N_cost):
            pk = p[k, :]
            yaw_k = th[k, 2]
            uk = np.array(self.solver.get(k, "u")).ravel()
            f_k = uk[0]

            e_p = pk - pref[k, :]
            e_y = yaw_k - yaw_ref[k]
            e_f = f_k - self.f_ref

            Jp[k] = float(e_p.T @ Wp @ e_p)
            Jy[k] = float(Wy * e_y * e_y)
            Jf[k] = float(Wf * e_f * e_f)
            Jtotal[k] = Jp[k] + Jy[k] + Jf[k]

        pN = p[N_cost, :]
        yawN = th[N_cost, 2]
        e_pN = pN - pref[-1, :]
        e_yN = yawN - yaw_ref[-1]
        Jp_N = float(e_pN.T @ Wp @ e_pN)
        Jy_N = float(Wy * e_yN * e_yN)
        Jtotal[N_cost] = Jp_N + Jy_N

        ax_cost   = fig.add_subplot(gs[2, 2])
        ax_cost.plot(t_stage, Jp, label="J_pos")
        ax_cost.plot(t_stage, Jy, label="J_yaw")
        ax_cost.plot(t_stage, Jf, label="J_thrust")
        ax_cost.plot(np.arange(N_cost + 1, dtype=np.float64) * dt, Jtotal, label="J_total", linewidth=2.0, alpha=0.3)
        ax_cost.set_title("Stage & Total cost")
        ax_cost.grid(True)
        ax_cost.legend(loc="upper right")

        #4 Solve-time history
        hist_size = self.solve_ms_hist.size
        count = min(self.solve_ms_count, hist_size)

        ax_ms   = fig.add_subplot(gs[3, 2])
        idx_end = self.solve_ms_count
        idx_start = max(0, idx_end - count)
        indices = [(i % hist_size) for i in range(idx_start, idx_end)]
        ms_vals = self.solve_ms_hist[indices]

        x_idx = np.arange(-count + 1, 1)
        ax_ms.plot(x_idx, ms_vals, marker="o")
        ax_ms.set_xlabel("index")
        ax_ms.set_ylabel("[ms]")
        ax_ms.set_title("Solve time(last {})".format(count))
        ax_ms.grid(True)

        # ------------------------------------------------------------------
        # 4th col (per-arm joint angles)
        # ------------------------------------------------------------------

        ax_q1 = fig.add_subplot(gs[0, 3])
        ax_q2 = fig.add_subplot(gs[1, 3])
        ax_q3 = fig.add_subplot(gs[2, 3])
        ax_q4 = fig.add_subplot(gs[3, 3])

        joint_colors = ["C0", "C1", "C2", "C3", "C4"]

        q1 = q[:, 0:5]
        q2 = q[:, 5:10]
        q3 = q[:, 10:15]
        q4 = q[:, 15:20]

        for j in range(5):
            ax_q1.plot(t, 180.0/np.pi*q1[:, j], color=joint_colors[j], label=f"joint{j+1}")
            ax_q2.plot(t, 180.0/np.pi*q2[:, j], color=joint_colors[j])
            ax_q3.plot(t, 180.0/np.pi*q3[:, j], color=joint_colors[j])
            ax_q4.plot(t, 180.0/np.pi*q4[:, j], color=joint_colors[j])

        ax_q1.set_title("Arm1 joint angles")
        ax_q2.set_title("Arm2 joint angles")
        ax_q3.set_title("Arm3 joint angles")
        ax_q4.set_title("Arm4 joint angles")

        ax_q1.set_ylabel("[deg]")
        ax_q2.set_ylabel("[deg]")
        ax_q3.set_ylabel("[deg]")
        ax_q4.set_ylabel("[deg]")
        ax_q4.set_xlabel("time [s]")

        ax_q1.legend(loc="upper right")

        ax_q1.grid(True)
        ax_q2.grid(True)
        ax_q3.grid(True)
        ax_q4.grid(True)

        # ------------------------------------------------------------------
        # Make 3D axes scale equal
        # ------------------------------------------------------------------
        xs = p[:, 0].copy()
        ys = p[:, 1].copy()
        zs = p[:, 2].copy()

        xs = np.concatenate([xs, g_p15_hist[:, 0], g_p25_hist[:, 0], g_p35_hist[:, 0], g_p45_hist[:, 0]])
        ys = np.concatenate([ys, g_p15_hist[:, 1], g_p25_hist[:, 1], g_p35_hist[:, 1], g_p45_hist[:, 1]])
        zs = np.concatenate([zs, g_p15_hist[:, 2], g_p25_hist[:, 2], g_p35_hist[:, 2], g_p45_hist[:, 2]])

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        z_min, z_max = zs.min(), zs.max()

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = 0.5 * max(x_range, y_range, z_range)

        mid_x = 0.5 * (x_min + x_max)
        mid_y = 0.5 * (y_min + y_max)
        mid_z = 0.5 * (z_min + z_max)

        for ax_3d in (ax_traj, ax_arm_traj):
            ax_3d.set_xlim3d(mid_x - max_range, mid_x + max_range)
            ax_3d.set_ylim3d(mid_y - max_range, mid_y + max_range)
            ax_3d.set_zlim3d(mid_z - max_range, mid_z + max_range)

        fig.tight_layout()

        #2 sliders
        init_t = 0
        joint_lines = []
        arm_colors = ["C1", "C2", "C3", "C4"]

        for arm_idx in range(4):
            gj = g_joints_hist[init_t, arm_idx, :, :]  # (6,3)
            line, = ax_arm_traj.plot(
                gj[:, 0], gj[:, 1], gj[:, 2],
                marker='o', markersize=2, linestyle='-', linewidth=1.0, alpha=0.8,
                color=arm_colors[arm_idx]
            )
            joint_lines.append(line)

        # slider axis
        bbox = ax_arm_traj.get_position()  # figure coordinates (0~1)
        slider_height = 0.03
        slider_margin = 0.01
        slider_width = bbox.width * 0.6
        slider_x0 = bbox.x0 + 0.5 * (bbox.width - slider_width)
        slider_y0 = max(0.0, bbox.y0 - slider_height - slider_margin)
        slider_ax = fig.add_axes([slider_x0, slider_y0, slider_width, slider_height])
        time_slider = Slider(ax=slider_ax, label="t [s]", valmin=0.0, valmax=(T - 1) * dt, valinit=0.0, valstep=dt,)
        time_slider.on_changed(update_slider)

        plt.show()

if __name__ == "__main__":
    # Debug run
    mpc = StriderNMPC()

    N   = mpc.N
    nx  = mpc.nx
    nu  = mpc.nu
    np_ = mpc.np

    # circular traj
    # r = 0.5
    # theta = np.linspace(-np.pi/2., np.pi/2., N)
    # x_traj = r*np.cos(theta)
    # y_traj = r + r*np.sin(theta)
    # z_traj = np.ones_like(theta) * 0.0    # constant altitude z = 1

    # linear traj
    lin = np.linspace(0.0, 1.0, N)
    x_traj = 0.0 * lin
    y_traj = lin
    z_traj = np.ones_like(lin) * 0.0    # constant altitude z = 1

    pos_d = np.zeros(3 * N, dtype=np.float64)
    for k in range(N): pos_d[3 * k : 3 * (k + 1)] = np.array([x_traj[k], y_traj[k], z_traj[k]], dtype=np.float64)

    yaw_d = np.zeros(N, dtype=np.float64)

    q_init_one_arm = np.array([
        0.0 * np.pi / 180.0,
        15.0 * np.pi / 180.0,
        60.0 * np.pi / 180.0,
        15.0 * np.pi / 180.0,
        0.0 * np.pi / 180.0,
    ], dtype=np.float64)
    q_init = np.tile(q_init_one_arm, 4)

    x_0 = np.zeros(nx, dtype=np.float64)
    x_0[8]=0.0 # yaw rad
    x_0[12:32] = q_init
    u_0 = np.zeros(nu, dtype=np.float64)
    u_0[0] = 50
    u_0[1:21] = q_init

    p = np.zeros(np_, dtype=np.float64)
    p[0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64) # delta
    p[12] = 1.0 * np.pi / 180.0 # theta_tilt = 0
    p[13] = 0.58 # l = 0.58

    # ----- set cot_R_b as Rz(alpha) -----
    alpha = 0.0 * np.pi / 180.0  # [rad], 원하는 각도
    c = np.cos(alpha)
    s = np.sin(alpha)
    Rz = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    p[3:12] = Rz.reshape(-1)      # body tilt cmd (cot_R_b = Rz)
    # ------------------------------------

    # Build mpci dict as C++ side would
    mpci = {
        "pos_d":  pos_d,
        "yaw_d":  yaw_d,
        "x_0":    x_0,
        "u_0":    u_0,
        "p":      p,
        "debug": True,
    }

    out = mpc.compute_MPC(mpci)

    mpc.print_last_debug()
    print("\n\n---------solved!---------\n\n")

    # print("[solved] MPC output u      :", out["u"])
    # print("[solved] MPC solve_ms      :", out["solve_ms"])
    # print("[solved] MPC solver status :", out["state"])