#!/usr/bin/env python3


import copy
import numpy as np

import matplotlib.pyplot as plt


from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry



class CBF():

    #Class Constructor
    def __init__(self):
        
           

        #Other definitions
        
        self.lidar_msg=[]
        self.steering_msg=Float32()
        self.throttle_msg=Float32()
               
        

        self.v = 0.0  # linear speed in x (longitudinal vehicle direction)
        


        self.steeringSaturation = 24 #degrees      


        self.maxThrotle=600 #% 0.24(experimentaly more than this cause crash)
            #Steeirng Control
        self.Ts=1/40 #s

        self.L=0.33 #[m] Wheelbase (distance from rear axis to the front axis)

        # Debug/inspection: last barrier interaction snapshot
        self.debug_barrier_interaction = True
        self.last_barrier_interaction = None
        self.debug_barrier_plot = True
        self.debug_barrier_plot_every = 10
        self.debug_barrier_plot_path = "/tmp/cbf_barrier_interaction.png"
        self._debug_barrier_plot_counter = 0
        self.k_td = 0.1
        self.min_accel = 0.5
        self.min_accel_speed = 0.5
        self.min_accel_hyst_on = 0.10
        self.min_accel_hyst_off = -0.10
        self._min_accel_active = False
        # Debug: print when CBF barrier becomes active
        self.debug_barrier_activation = True
        self.debug_barrier_activation_every = 1
        self._debug_barrier_activation_counter = 0


    def __setStatus(self, LiDAR_beams,steer = 0.0, throttle = 0.0, speed= 0.0):
        self.lidar_msg = self.lidarPreprocessing(LiDAR_beams)
        self.steering_msg = steer
        self.throttle_msg = throttle
        self.v = speed

    #=============================
    # LiDAR Processing Methods
    # ============================  
      
   
    
    def lidarPreprocessing(self, lidar):        
        for i in range(len(lidar)):
            if lidar[i] != lidar[i] or lidar[i] == float('inf') or lidar[i] == float('-inf'):
                #lidar.ranges[i] = max_val
                lidar[i] = 10.0 #lidar.range_max
        
        return lidar

  

    def __publish_control(self, steer, throttle):
       
        #Saturate steering u=theta
        max_steer = np.radians(self.steeringSaturation)  
        min_steer = - max_steer  

        # Saturate steer value 
        steer = max(min(steer, max_steer), min_steer)*1.0 #cast to Float

        #Saturate Max Throttle
        if throttle>self.maxThrotle:
            throttle=self.maxThrotle

        return {'steer':steer, 'acceleration':throttle}

    #=============================
    # QP Hildreth Solver Code
    # ============================   
    def qp_hildreth(self,H, f, A, b, max_iter=38, tol=1e-8):
        """
        Solves:
            min 0.5 x^T H x + f^T x
            s.t. A x <= b
        """

        n_constraints = A.shape[0]

        # Unconstrained solution
        eta = -np.linalg.solve(H, f)

        # Check feasibility
        violations = A @ eta - b
        if np.all(violations <= 0):
            print("Control Aplicado: Nominal")
            return eta

        # Hildreth method
        H_inv = np.linalg.inv(H)
        P = A @ H_inv @ A.T
        d = A @ H_inv @ f + b

        lambda_ = np.zeros(n_constraints)

        for _ in range(max_iter):
            lambda_prev = lambda_.copy()

            for i in range(n_constraints):
                w = P[i, :] @ lambda_ - P[i, i] * lambda_[i]
                w += d[i]

                #lambda_[i] = max(0.0, -w / P[i, i])
                if P[i, i] > 1e-12:
                    lambda_[i] = max(0.0, -w / P[i, i]) #to avoid dividing by zero


            if np.linalg.norm(lambda_ - lambda_prev)**2 < tol:
                break

        eta = -H_inv @ f - H_inv @ A.T @ lambda_
        print("Control Aplicado: CBF")
        return eta
 
         

    #=============================
    # CBF Code
    # ============================      
    def __barrier_interaction(self, Td, gamma, h, v, psi, R, Dmin, cbf_condition=None):
        """
        Build a diagnostic snapshot of how Td and gamma affect h and alpha_h.
        Stores the snapshot in self.last_barrier_interaction and returns it.
        """
        # Contribution of Td term in h
        td_term = Td * v * np.cos(psi)
        h_reconstructed = R - td_term - Dmin
        alpha_h = gamma * h

        if cbf_condition is not None:
            cbf_condition = np.array(cbf_condition, copy=False)
            cbf_violation = -cbf_condition  # A_q*u_nominal - b_q
        else:
            cbf_violation = None

        stats = {
            "h_min": float(np.min(h)),
            "h_max": float(np.max(h)),
            "alpha_h_min": float(np.min(alpha_h)),
            "alpha_h_max": float(np.max(alpha_h)),
            "Td_min": float(np.min(Td)),
            "Td_max": float(np.max(Td)),
            "gamma_min": float(np.min(gamma)),
            "gamma_max": float(np.max(gamma)),
        }
        if cbf_condition is not None:
            stats["cbf_condition_min"] = float(np.min(cbf_condition))
            stats["cbf_condition_max"] = float(np.max(cbf_condition))

        snapshot = {
            "Td": Td,
            "gamma": gamma,
            "h": h,
            "alpha_h": alpha_h,
            "cbf_condition": cbf_condition,
            "cbf_violation": cbf_violation,
            "h_components": {
                "range": R,
                "td_term": td_term,
                "Dmin": Dmin,
            },
            "h_reconstructed": h_reconstructed,
            "stats": stats,
        }

        self.last_barrier_interaction = snapshot

        if self.debug_barrier_interaction:
            i_min = int(np.argmin(h))
            # print(
            #     "Barrier interaction: "
            #     f"h[{stats['h_min']:.3f},{stats['h_max']:.3f}] "
            #     f"alpha_h[{stats['alpha_h_min']:.3f},{stats['alpha_h_max']:.3f}] "
            #     f"Td[{stats['Td_min']:.3f},{stats['Td_max']:.3f}] "
            #     f"gamma[{stats['gamma_min']:.3f},{stats['gamma_max']:.3f}]"
            # )
            # print(
            #     "Barrier min ray: "
            #     f"i={i_min} "
            #     f"psi_deg={np.degrees(psi[i_min]):.1f} "
            #     f"R={R[i_min]:.3f} "
            #     f"Td={Td[i_min]:.3f} "
            #     f"h={h[i_min]:.3f} "
            #     f"td_term={td_term[i_min]:.3f} "
            #     f"Dmin={Dmin:.3f}"
            # )

        if self.debug_barrier_plot:
            self._debug_barrier_plot_counter += 1
            if self._debug_barrier_plot_counter % self.debug_barrier_plot_every == 0:
                try:
                    idx = np.arange(h.size)
                    plt.figure(100)
                    plt.clf()
                    n_rows = 3 if cbf_condition is None else 4
                    ax1 = plt.subplot(n_rows, 2, 1)
                    ax1.plot(idx, h)
                    ax1.axhline(0.0, color="k", linewidth=0.8)
                    ax1.set_title("h")
                    ax1.set_xticks(np.round(np.linspace(idx.min(), idx.max(), 7)).astype(int))
                    ax1.set_xticklabels([str(int(x)) for x in ax1.get_xticks()])

                    ax2 = plt.subplot(n_rows, 2, 2)
                    ax2.plot(idx, alpha_h, color="tab:red")
                    ax2.set_title("alpha_h = gamma*h")
                    ax2.set_xticks(np.round(np.linspace(idx.min(), idx.max(), 7)).astype(int))
                    ax2.set_xticklabels([str(int(x)) for x in ax2.get_xticks()])

                    ax3 = plt.subplot(n_rows, 2, 3)
                    ax3.plot(idx, Td, color="tab:green")
                    ax3.set_title("Td")
                    ax3.set_xticks(np.round(np.linspace(idx.min(), idx.max(), 7)).astype(int))
                    ax3.set_xticklabels([str(int(x)) for x in ax3.get_xticks()])

                    ax4 = plt.subplot(n_rows, 2, 4)
                    ax4.plot(idx, gamma, color="tab:purple")
                    ax4.set_title("gamma")
                    ax4.set_xticks(np.round(np.linspace(idx.min(), idx.max(), 7)).astype(int))
                    ax4.set_xticklabels([str(int(x)) for x in ax4.get_xticks()])

                    ax5 = plt.subplot(n_rows, 2, 5)
                    ax5.plot(np.degrees(psi), R, color="tab:blue")
                    ax5.set_title("LiDAR range R vs psi")
                    ax5.set_xticks(np.round(np.linspace(np.degrees(psi).min(), np.degrees(psi).max(), 7)).astype(int))
                    ax5.set_xticklabels([str(int(x)) for x in ax5.get_xticks()])

                    ax6 = plt.subplot(n_rows, 2, 6)
                    ax6.plot(idx, np.degrees(psi), color="tab:orange")
                    ax6.set_title("LiDAR angle psi (deg)")
                    ax6.set_xticks(np.round(np.linspace(idx.min(), idx.max(), 7)).astype(int))
                    ax6.set_xticklabels([str(int(x)) for x in ax6.get_xticks()])

                    if cbf_condition is not None:
                        ax7 = plt.subplot(n_rows, 2, 7)
                        ax7.plot(idx, cbf_condition, color="tab:brown")
                        ax7.axhline(0.0, color="k", linewidth=0.8)
                        ax7.set_title("CBF condition (>=0)")
                        ax7.set_xticks(np.round(np.linspace(idx.min(), idx.max(), 7)).astype(int))
                        ax7.set_xticklabels([str(int(x)) for x in ax7.get_xticks()])

                        ax8 = plt.subplot(n_rows, 2, 8)
                        ax8.plot(idx, cbf_violation, color="tab:gray")
                        ax8.axhline(0.0, color="k", linewidth=0.8)
                        ax8.set_title("CBF violation = A_q*u - b_q")
                        ax8.set_xticks(np.round(np.linspace(idx.min(), idx.max(), 7)).astype(int))
                        ax8.set_xticklabels([str(int(x)) for x in ax8.get_xticks()])
                    plt.tight_layout()
                    plt.savefig(self.debug_barrier_plot_path, dpi=150)
                except Exception as exc:
                    print(f"Barrier plot failed: {exc}")

        return snapshot

    def __cbf_filter(self,v, scan_ranges, scan_angles, u_nominal):
        """
        Parameters
        ----------
        v : float
            Vehicle linear velocity
        scan_ranges : np.ndarray (N,)
            LiDAR ranges
        scan_angles : np.ndarray (N,)
            Corresponding beam angles
        u_nominal : np.ndarray (2,)
            Nominal control [acceleration, angular_speed]

        Returns
        -------
        u_opt : np.ndarray (2,)
            CBF-filtered control
        h : np.ndarray (nBeams,)
            CBF values
        """

        # =================
        # LiDAR data
        # =================
        scan_max_range = 25
        n_beams = 30

        N = scan_angles.size
        idx = np.linspace(0, N - 1, n_beams)

        psi = np.interp(idx, np.arange(N), scan_angles)
        R = np.interp(idx, np.arange(N), scan_ranges)
        R = np.nan_to_num(R, nan=scan_max_range)

        # Static obstacles
        v_obj = np.zeros(n_beams)

        # =================
        # CBF parameters
        # =================
        #Td = 0.5 * np.cos(psi) + 0.1
        #Td = 0.75 *(np.cos(psi))+0.25
        Td = 0.01*(0.5 * np.cos(psi) + 0.5)
        #Td = np.clip(Td, 0.2, 0.8)

        gamma = 20.0 + 5.0 * np.abs(np.sin(psi))       
        

        # =================
        # Nonlinear model
        # =================
        h = np.zeros(n_beams)
        alpha_h = np.zeros(n_beams)
        Lfh = np.zeros(n_beams)
        Lgh = np.zeros((n_beams, 2))

        for i in range(n_beams):
            Lfh[i] = v_obj[i] - v * np.cos(psi[i]) - Td[i]*(v**2)*(np.sin(psi[i])**2)/R[i]
            Lgh[i, :] = [
                -Td[i] * np.cos(psi[i]),
                Td[i] * (v**2) * np.sin(psi[i])/self.L
            ]

            Dmin = 0.05

            h[i] = R[i] - Td[i] * v * np.cos(psi[i]) - Dmin
            alpha_h[i] = gamma[i] * h[i]#**3

        cbf_condition = Lgh @ u_nominal + Lfh + alpha_h

        # Store a diagnostic snapshot of Td/gamma interaction with h
        self.__barrier_interaction(Td, gamma, h, v, psi, R, Dmin, cbf_condition=cbf_condition)

        # =========================
        # Quadratic Program
        # =========================
        #u_min = np.array([-3.0, -0.4])  #3 is in m/s^2 because u1=a
        #u_max = np.array([ 3.0,  +0.4]) #0.4 is approximatly the tan(20º) because u2=tan(delta), delta=steering

        u2_max = np.tan(np.radians(self.steeringSaturation))
        u_min = np.array([-5.0, -u2_max])  # u2 = tan(delta)
        u_max = np.array([ 5.0,  +u2_max])
        A_lim = np.array([
            [ 1,  0],
            [-1,  0],
            [ 0,  1],
            [ 0, -1]
        ])

        b_lim = np.array([
            u_max[0],
            -u_min[0],
            u_max[1],
            -u_min[1]
        ])

        H = np.array([
            [2.0,    0.0],
            [   0.0,    2.0]
        ])

        f = -2.0 * u_nominal

        A_q = -Lgh
        b_q = Lfh + alpha_h
        b_q_min = float(np.min(b_q))

        # Debug: report which physical parameter triggered the barrier
        if self.debug_barrier_activation:
            violation_q = A_q @ u_nominal - b_q
            max_violation = float(np.max(violation_q))
            if max_violation > 1e-6:
                self._debug_barrier_activation_counter += 1
                if self._debug_barrier_activation_counter % self.debug_barrier_activation_every == 0:
                    i_v = int(np.argmax(violation_q))
                    td_term_i = Td[i_v] * v * np.cos(psi[i_v])
                    print("Barrera CBF activa: el control nominal viola la restricción.")
                    print(
                        "  Parámetro físico (rayo crítico): "
                        f"i={i_v}, psi={np.degrees(psi[i_v]):.1f} deg, "
                        f"R={R[i_v]:.3f} m, v={v:.3f} m/s"
                    )
                    print(
                        "  h = R - Td*v*cos(psi) - Dmin: "
                        f"{h[i_v]:.3f} = {R[i_v]:.3f} - {td_term_i:.3f} - {Dmin:.3f}"
                    )
                    cbf_condition = (
                        Lgh[i_v, 0] * u_nominal[0]
                        + Lgh[i_v, 1] * u_nominal[1]
                        + Lfh[i_v]
                        + alpha_h[i_v]
                    )
                    term_u1 = Lgh[i_v, 0] * u_nominal[0]
                    term_u2 = Lgh[i_v, 1] * u_nominal[1]
                    term_lfh = Lfh[i_v]
                    term_alpha = alpha_h[i_v]
                    print(
                        "  Condición CBF (debe ser >= 0): "
                        f"Lgh*u + Lfh + alpha_h = {cbf_condition:.3f}"
                    )
                    print(
                        "  Desglose: "
                        f"Lgh1*u1={term_u1:.3f} + Lgh2*u2={term_u2:.3f} "
                        f"+ Lfh={term_lfh:.3f} + alpha_h={term_alpha:.3f}"
                    )
                    print(
                        f"  Td={Td[i_v]:.3f}, gamma={gamma[i_v]:.3f}, "
                        f"alpha_h={alpha_h[i_v]:.3f}, b_q={b_q[i_v]:.3f}, "
                        f"viol={max_violation:.3f}"
                    )

        A = np.vstack((A_q, A_lim))
        b = np.hstack((b_q, b_lim))

        #DEBUG
        bq_min = np.min(b_q)
        i_bq = np.argmin(b_q)
        # print(f"b_q_min={bq_min:.3f} at i={i_bq} psi={np.degrees(psi[i_bq]):.1f}")
        # print(f"u_nominal={u_nominal}")


        # =========================
        # Solve QP using Hildreth
        # =========================
        u_opt = self.qp_hildreth(H, f, A, b)

        # Expose min CBF slack for downstream decisions
        self.last_bq_min = b_q_min

        return u_opt, h

    
   

    def control(self,LiDAR_beams,steer = 0.0, throttle = 0.0, speed= 0.0):

        self.__setStatus(LiDAR_beams,steer, throttle, speed)
        #1) acceleration
        a=throttle
        
        #2) angular speed
        delta=steer

        u1=a
        u2=np.tan(delta)

        #3) nominal control
        u_nominal=np.array([u1,u2])
        
        #4) LiDAR msg extraction
        if len(self.lidar_msg) == 0:
            return #avoid crashing due to empty LiDAR msg
        
        scan_ranges = np.array(self.lidar_msg)
        angle_min = 0.0#-np.pi/4 #self.lidar_msg.angle_min
        angle_increment = np.deg2rad(0.25)#0.004363323 #(0.25 grad) 
        scan_angles = angle_min + np.arange(scan_ranges.size) * angle_increment

        # LiDAR frame: 0 rad points to the right; CBF expects 0 rad at the front.
        # Shift angles by -pi/2 and wrap to [-pi, pi].
        scan_angles = scan_angles - (np.pi / 2.0)
        scan_angles = (scan_angles + np.pi) % (2.0 * np.pi) - np.pi

        #5) Crop LiDAR info in [-pi/2,+pi/2]
        # Desired angular window
        angle_low  =  -np.deg2rad(30)#-np.pi/ 2
        angle_high =  np.deg2rad(30)#np.pi/ 2
        # Boolean mask
        mask = (scan_angles >= angle_low) & (scan_angles <= angle_high)
        # Crop both arrays consistently
        scan_angles_cropped = scan_angles[mask]
        scan_ranges_cropped = scan_ranges[mask]

        #6) Speed from VESC odometry
        v=self.v
        #7) Apply CBF Filter
        [u_opt, h]=self.__cbf_filter(v, scan_ranges_cropped, scan_angles_cropped, u_nominal) 
        #8) Send filtered control to vehicle
        a=u_opt[0] #u1
        tan_delta=u_opt[1] #u2        

        #9) Transform u2 in delta (steering)
        delta = np.arctan(tan_delta)

        # If we are nearly stopped, avoid deadband by enforcing a minimum accel
        # Only force min accel when CBF constraints are not violated
        # bq_min = getattr(self, "last_bq_min", 0.0)
        # if bq_min >= self.min_accel_hyst_on:
        #     self._min_accel_active = True
        # elif bq_min <= self.min_accel_hyst_off:
        #     self._min_accel_active = False

        # if self._min_accel_active and u_nominal[0] > 0.05 and v < self.min_accel_speed:
        #     a = max(a, self.min_accel)

        # print(
        #     f"FTG: Acc={a:.3f} m/s^2, Steering={np.degrees(delta):.3f} º"
        # )

        return self.__publish_control(steer=delta , throttle=a)
        
        

        
        

        
