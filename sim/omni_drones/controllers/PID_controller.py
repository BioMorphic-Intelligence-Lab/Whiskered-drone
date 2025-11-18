import math

class PID:
    def __init__(self, kp, ki, kd, kff=0.0, dt=0.01, integral_limit=None, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kff = kff
        self.dt = dt
        self.integral_limit = integral_limit
        self.output_limit = output_limit

        self.integral = 0.0
        self.last_error = 0.0

    def reset(self, current_value=0.0):
        self.integral = 0.0
        self.last_error = 0.0

    def update(self, actual, desired, feedforward=0.0):
        error = desired - actual
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

        derivative = (error - self.last_error) / self.dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative) + (self.kff * feedforward)

        if self.output_limit is not None:
            output = max(min(output, self.output_limit), -self.output_limit)

        self.last_error = error
        return output


class PositionController:
    def __init__(self, dt=0.01):
        self.pid_x = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)
        self.pid_y = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)
        self.pid_z = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)

        self.pid_vx = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)
        self.pid_vy = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)
        self.pid_vz = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)

    def update(self, pos_actual, vel_actual, vel_desired, thrust_base=0.5):
        # 目标速度为输入，位置误差作为辅助信息可以后续加
        # 速度误差控制生成加速度期望
        ax_des = self.pid_vx.update(vel_actual[0], vel_desired[0])
        ay_des = self.pid_vy.update(vel_actual[1], vel_desired[1])
        az_des = self.pid_vz.update(vel_actual[2], vel_desired[2])

        # 转换成期望姿态角（简化，roll和pitch估算）
        pitch_des = math.atan2(ax_des, 9.81)
        roll_des = -math.atan2(ay_des, 9.81)

        thrust = thrust_base + az_des

        return (roll_des, pitch_des, thrust)


class AttitudeController:
    def __init__(self, dt=0.01):
        self.pid_roll_rate = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)
        self.pid_pitch_rate = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)
        self.pid_yaw_rate = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)

        self.pid_roll = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)
        self.pid_pitch = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)
        self.pid_yaw = PID(kp=1.0, ki=0.0, kd=0.1, dt=dt)

    def update(self, euler_actual, euler_desired, rate_actual):
        roll_rate_des = self.pid_roll.update(euler_actual[0], euler_desired[0])
        pitch_rate_des = self.pid_pitch.update(euler_actual[1], euler_desired[1])
        yaw_rate_des = self.pid_yaw.update(euler_actual[2], euler_desired[2])

        roll_output = self.pid_roll_rate.update(rate_actual[0], roll_rate_des)
        pitch_output = self.pid_pitch_rate.update(rate_actual[1], pitch_rate_des)
        yaw_output = self.pid_yaw_rate.update(rate_actual[2], yaw_rate_des)

        return (roll_output, pitch_output, yaw_output)

class SimplePIDController:
    def __init__(self, dt=0.01):
        self.position_controller = PositionController(dt)
        self.attitude_controller = AttitudeController(dt)

    def compute(self, root_state, target_pos=None, target_vel=None, target_acc=None, target_yaw=None, body_rate=False):
        # root_state 包含 position, velocity, euler_angles, angular_rates (dict 或 tensor)
        # 兼容性：都用 dict 方便，也可用 tensor，需改写

        # 默认值
        if target_pos is None:
            target_pos = root_state['position']
        if target_vel is None:
            target_vel = [0, 0, 0]
        if target_acc is None:
            target_acc = [0, 0, 0]
        if target_yaw is None:
            target_yaw = root_state['euler_angles'][2]

        # 这里只用速度目标做演示
        roll_des, pitch_des, thrust = self.position_controller.update(
            pos_actual=root_state['position'],
            vel_actual=root_state['velocity'],
            vel_desired=target_vel,
            thrust_base=0.5
        )

        roll_output, pitch_output, yaw_output = self.attitude_controller.update(
            euler_actual=root_state['euler_angles'],
            euler_desired=(roll_des, pitch_des, target_yaw),
            rate_actual=root_state['angular_rates']
        )

        # 返回类似 cmd 格式，但本示例只是姿态控制量和推力
        return {
            'roll_output': roll_output,
            'pitch_output': pitch_output,
            'yaw_output': yaw_output,
            'thrust': thrust
        }
