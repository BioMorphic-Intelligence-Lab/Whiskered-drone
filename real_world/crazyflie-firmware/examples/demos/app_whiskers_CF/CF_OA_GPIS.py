import logging
import sys
import time
import os
import argparse
from datetime import datetime
from pathlib import Path
from threading import Thread
import motioncapture

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from FileLogger import FileLogger
from cflib.utils import uri_helper
import math

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# The host name or ip address of the mocap system
host_name = '192.168.209.81'

# The type of the mocap system
# Valid options are: 'vicon', 'optitrack', 'optitrack_closed_source', 'qualisys', 'nokov', 'vrpn', 'motionanalysis'
# mocap_system_type = 'optitrack'

# body_name = "WhiskeredDrone"  


# # True: send position and orientation; False: send position only
# send_full_pose = True

# #obtain position and rotation from mocap
# class MocapWrapper(Thread):
#     def __init__(self, body_name, file_logger):
#         super().__init__()
#         self.body_name = body_name
#         self.file_logger = file_logger
#         self.on_pose = None  
#         self._stay_open = True
#         self.start()

#     def quaternion_to_yaw(self, x, y, z, w):
#         """从四元数计算 yaw"""
#         yaw = math.atan2(2.0 * (w * y - z * x), 1.0 - 2.0 * (x * x + y * y))
#         return yaw

#     def run(self):
#         mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
#         while self._stay_open:
#             mc.waitForNextFrame()
#             for name, obj in mc.rigidBodies.items():
#                 if name == self.body_name:
#                     if self.on_pose:
#                         pos = obj.position
#                         rot = obj.rotation
#                         yaw = self.quaternion_to_yaw(rot.x, rot.y, rot.z, rot.w)
#                         self.on_pose({
#                             "pos_x": pos[0],
#                             "pos_y": pos[2],
#                             "yaw": yaw
#                         })

def unlock_drone(cf):
    cf.param.set_value('app.stateOuterLoop', '1')

def stop_drone(cf):
    cf.param.set_value('app.stateOuterLoop', '2')

def apply_mlp(cf):
    cf.param.set_value('app.statemlp', '1')
def apply_kf(cf):
    cf.param.set_value('app.statekf', '1')
def data_collection_mode(cf):
    cf.param.set_value('app.statemlp', '2')

def exploration_mode(cf):
    cf.param.set_value('app.stateexp', '1')

def exploration_mode(cf):
    cf.param.set_value('app.stateexp', '1')

def exploration_mode_gpis(cf):
    cf.param.set_value('app.stategpis', '1')

def set_initial_params(cf, MIN_THRESHOLD1, MAX_THRESHOLD1, MIN_THRESHOLD2, MAX_THRESHOLD2, maxSpeed, maxTurnRate):
    cf.param.set_value('app.MIN_THRESHOLD1', str(MIN_THRESHOLD1)) 
    cf.param.set_value('app.MAX_THRESHOLD1', str(MAX_THRESHOLD1))
    cf.param.set_value('app.MIN_THRESHOLD2', str(MIN_THRESHOLD2))
    cf.param.set_value('app.MAX_THRESHOLD2', str(MAX_THRESHOLD2))
    cf.param.set_value('app.maxSpeed', str(maxSpeed))
    cf.param.set_value('app.maxTurnRate', str(maxTurnRate))

def post_cal_params(cf, scale_1, scale_2, offset_1, offset_2):
    cf.param.set_value('app.scale_1', str(scale_1)) 
    cf.param.set_value('app.scale_2', str(scale_2))
    cf.param.set_value('app.offset_1', str(offset_1))
    cf.param.set_value('app.offset_2', str(offset_2))

def get_filename():
    fileroot = args["fileroot"] 
        
    if args["filename"] is not None:
        name = args["filename"] + ".csv"
        fname = os.path.normpath(os.path.join(
            os.getcwd(), fileroot, name))
        i = 0
        while os.path.isfile(fname):
            i = i + 1
            name = args["filename"] + "_" + str(i) + ".csv"
            fname = os.path.normpath(os.path.join(
                os.getcwd(), fileroot, name))

    else:
        # get relevant arguments
        # keywords = args["keywords"]
        # estimator = args["estimator"]
        # uwb = args["uwb"]
        # optitrack = args["optitrack"]
        # trajectory = args["trajectory"]

        # Date
        date = datetime.today().strftime(r"%Y-%m-%d+%H:%M:%S")

        # Additional keywords
        # if keywords is not None:
        #     keywords = "+" + "+".join(keywords)
        # else:
        #     keywords = ""

        # Options
        # if optitrack == "logging":
        #     options = f"optitracklog"
        # elif optitrack == "state":
        #     options = f"optitrackstate"
        # else:
        options = f""

        # Join
        name = "{}+{}.csv".format(date, options)
        fname = os.path.normpath(os.path.join(os.getcwd(), fileroot, name))
    return fname

def setup_logger():
    # Create directory if not there
    Path(args["fileroot"]).mkdir(exist_ok=True)
        
    # Create filename from options and date
    log_file = get_filename()

    print(f"Log location: {log_file}")


    # Logger setup
    logconfig = args["logconfig"]
    flogger = FileLogger(cf, logconfig, log_file)
    # Enable log configurations based on system setup:
    # Defaults
    # flogger.enableConfig("attitude")
    # flogger.enableConfig("gyros")
    # flogger.enableConfig("acc")
    flogger.enableConfig("state")
    flogger.enableConfig("whisker1")
    flogger.enableConfig("whisker2")
    flogger.enableConfig("PreWhisker1")
    flogger.enableConfig("PreWhisker2")
    flogger.enableConfig("MLPOUTPUT")
    # flogger.enableConfig("StateOuterLoop")
    flogger.enableConfig("orientation")
    flogger.enableConfig("laser")
    flogger.enableConfig("GPISLABEL")
    flogger.enableConfig("KFOUTPUT")

    # flogger.enableConfig("mocap")

    # # enable mocap
    # mocap_logger = MocapWrapper(body_name, flogger)
    
    # def mocap_callback(data):
    #     flogger.registerData("mocap", data)

    # # callback
    # mocap_logger.on_pose = mocap_callback

    # start logger
    flogger.start()
    print(1)
    print("Logging started with callback.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fileroot", type=str, required=True)
    parser.add_argument("--logconfig", type=str, required=True)
    parser.add_argument("--filename", type=str, default=None)
    args = vars(parser.parse_args())
    
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers()

    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        cf.platform.send_arming_request(True)
        filelogger=setup_logger()
        keep_flying = True
        set_initial_params(scf.cf, 50.0, 90.0, 50.0, 90.0, 0.2, 25.0)
        post_cal_params(scf.cf, 1.00, 1.00, 0.00, 0.00)
        time.sleep(3)
        unlock_drone(scf.cf)
        apply_mlp(scf.cf)
        apply_kf(scf.cf)
        exploration_mode(scf.cf)
        exploration_mode_gpis(scf.cf)
        print("start flying!")
        try:
            while keep_flying:
                command = input("Enter 's' to stop the drone: ").strip().lower()
                if command == 's':
                    print("Stop command received!")
                    keep_flying = False

            while not keep_flying:
                stop_drone(scf.cf) 
                print("Stopping the drone...")
                time.sleep(0.1) 
        except KeyboardInterrupt:
            print('Demo terminated!')
