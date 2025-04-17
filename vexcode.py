#region VEXcode Generated Robot Configuration
from vex import *
import urandom

# Brain should be defined by default
brain = Brain()

# Robot configuration code
brain_inertial = Inertial()
left_drive_smart = Motor(Ports.PORT6, False)
right_drive_smart = Motor(Ports.PORT10, True)
drivetrain = SmartDrive(left_drive_smart, right_drive_smart, brain_inertial, 259.34, 320, 40, MM, 1)
armeture = Motor(Ports.PORT3, False)
claw = Motor(Ports.PORT4, False)
digital_in_d = DigitalIn(brain.three_wire_port.d)
digital_in_g = DigitalIn(brain.three_wire_port.g)


# Wait for sensor(s) to fully initialize
wait(100, MSEC)

# generating and setting random seed
def initializeRandomSeed():
    wait(100, MSEC)
    xaxis = brain_inertial.acceleration(XAXIS) * 1000
    yaxis = brain_inertial.acceleration(YAXIS) * 1000
    zaxis = brain_inertial.acceleration(ZAXIS) * 1000
    systemTime = brain.timer.system() * 100
    urandom.seed(int(xaxis + yaxis + zaxis + systemTime)) 

# Initialize random seed 
initializeRandomSeed()

vexcode_initial_drivetrain_calibration_completed = False
def calibrate_drivetrain():
    # Calibrate the Drivetrain Inertial
    global vexcode_initial_drivetrain_calibration_completed
    sleep(200, MSEC)
    brain.screen.print("Calibrating")
    brain.screen.next_row()
    brain.screen.print("Inertial")
    brain_inertial.calibrate()
    while brain_inertial.is_calibrating():
        sleep(25, MSEC)
    vexcode_initial_drivetrain_calibration_completed = True
    brain.screen.clear_screen()
    brain.screen.set_cursor(1, 1)


# Calibrate the Drivetrain
calibrate_drivetrain()

#endregion VEXcode Generated Robot Configuration
# ------------------------------------------
# 
# 	Project:      VEXcomm
#	Author:       Ethan
#	Created:
#	Description:  Pi to Vex communication
# 
# ------------------------------------------

# Library imports
from vex import *
import time

# communication commands
com_stop = 0
com_forward = 1
com_reverse = 2
com_turn_left = 3
com_turn_right = 4
com_turn_left_angle = 5
com_turn_right_angle = 6
com_forward_for = 7
com_reverse_for = 8
com_arm_up_for = 9
com_arm_down_for = 10
com_arm_up = 11
com_arm_down = 12
com_claw_open = 13
com_claw_close = 14
com_drive_velocity = 15
com_claw_velocity = 16
com_arm_velocity = 17
com_arm_position = 18

brain.screen.print("running")

def vcomm():
    brain.screen.set_font(FontType.MONO12)
    claw.set_max_torque(100,PERCENT)
    armeture.spin_to_position(40,DEGREES)
    armeture.set_velocity(100,PERCENT)
    byt = int(0)
    bit_count = 0
    operation = 0
    data = bytearray()
    while True:
        #time.sleep(0.002) # 2ms
        if digital_in_g.value():
            s = digital_in_d.value()
            byt = byt >> 1 # Shift right to make room for the new bit
            if s:
                byt = byt | 0x80 # set the MSB to 1 if s is True
            bit_count = bit_count + 1
            if bit_count == 8:
                brain.screen.clear_screen()
                brain.screen.set_cursor(1,1)
                brain.screen.print("recv")
                brain.screen.print(bit_count)
                brain.screen.next_row()
                brain.screen.print(byt)
                brain.screen.next_row()
                brain.screen.print(time.time())
                brain.screen.next_row()
                if byt == com_stop: #0
                    drivetrain.stop()
                    claw.stop()
                    armeture.stop()
                    armeture.spin_to_position(40,DEGREES,wait=False)
                    bit_count = 0
                    byt = 0
                if byt == com_forward: #1
                    drivetrain.drive(FORWARD)
                    bit_count = 0
                    byt = 0
                if byt == com_reverse: #2
                    drivetrain.drive(REVERSE)
                    bit_count = 0
                    byt = 0
                if byt == com_turn_left: #3
                    drivetrain.turn(LEFT)
                    bit_count = 0
                    byt = 0
                if byt == com_turn_right: #4
                    drivetrain.turn(RIGHT)
                    bit_count = 0
                    byt = 0
                if byt == com_turn_left_angle: #5
                    data.append(byt)
                    byt = 0
                if byt == com_turn_right_angle: #6
                    data.append(byt)
                    byt = 0
                if byt == com_forward_for: # 7
                    data.append(byt)
                    byt = 0
                if byt == com_reverse_for: # 8
                    data.append(byt)
                    byt = 0
                if byt == com_arm_up_for: # 9
                    data.append(byt)
                    byt = 0
                if byt == com_arm_down_for: # 10
                    data.append(byt)
                    byt = 0
                if byt == com_arm_up: # 11
                    armeture.spin(FORWARD)
                    bit_count = 0
                    byt = 0
                if byt == com_arm_down: # 12
                    armeture.spin(REVERSE)
                    bit_count = 0
                    byt = 0
                if byt == com_claw_open: # 13
                    claw.spin(FORWARD)
                    bit_count = 0
                    byt = 0
                if byt == com_claw_close: # 14
                    claw.spin(REVERSE)
                    bit_count = 0
                    byt = 0
                if byt == com_drive_velocity: # 15
                    data.append(byt)
                    byt = 0
                if byt == com_claw_velocity: # 16
                    data.append(byt)
                    byt = 0
                if byt == com_arm_velocity: # 17
                    data.append(byt)
                    byt = 0
                if byt == com_arm_position: # 18
                    data.append(byt)
                    byt = 0
            if bit_count == 16: # 2 bytes recieved, 1st is command 2nd is upper 8 bits of 16 bits
                data.append(byt)
                byt = 0
            if bit_count == 24: # all 3 bytes recieved
                brain.screen.clear_screen()
                brain.screen.set_cursor(0,0)
                brain.screen.next_row()
                brain.screen.print("24 bits recv")
                brain.screen.next_row()
                brain.screen.print(int.from_bytes(data[1:3],'big'))
                data.append(byt)
                byt = 0
                bit_count = 0
                print(len(data))
                # lots of repeated code because it makes changing things easier and reduces pesky bugs and logic errors
                if len(data) > 1: # needed because when we reset bytearray index is nil
                    if data[0] == com_turn_left_angle: # 5
                        drivetrain.turn_for(LEFT, int.from_bytes(data[1:3],'big'),DEGREES,wait=False)
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_turn_right_angle:
                        drivetrain.turn_for(RIGHT, int.from_bytes(data[1:3],'big'),DEGREES,wait=False)
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_forward_for:
                        drivetrain.drive_for(FORWARD, int.from_bytes(data[1:3],'big'),MM,wait=False)
                        brain.screen.clear_screen()
                        brain.screen.set_cursor(1,1)
                        brain.screen.print("dst")
                        brain.screen.next_row()
                        brain.screen.print(int.from_bytes(data[1:3],'big'))
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_reverse_for:
                        drivetrain.drive_for(REVERSE, int.from_bytes(data[1:3],'big'),MM,wait=False)
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_arm_up_for:
                        armeture.spin_to_position(int.from_bytes(data[1:3],'big'),DEGREES,wait=False)
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_arm_down_for:
                        armeture.spin_to_position(-int.from_bytes(data[1:3],'big'),DEGREES,wait=False)
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_drive_velocity:
                        drivetrain.set_velocity(int.from_bytes(data[1:3],'big'),PERCENT)
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_claw_velocity:
                        claw.set_velocity(int.from_bytes(data[1:3],'big'),PERCENT)
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_arm_velocity:
                        armeture.set_velocity(int.from_bytes(data[1:3],'big'),PERCENT)
                        data = bytearray()
                if len(data) > 1:
                    if data[0] == com_arm_position:
                        armeture.spin_to_position(int.from_bytes(data[1:3],'big'),DEGREES,wait=False)
                        data = bytearray()
            while True: # hold until clock signal goes low
                if not digital_in_g.value():
                    time.sleep(0.005) # sleep for 5ms, we don't need to spin cycles
                    break

vcomm()