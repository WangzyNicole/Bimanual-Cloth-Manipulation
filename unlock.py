from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

PORT1 = "/dev/cu.usbmodem5AE60581371"
PORT2 = "/dev/cu.usbmodem5AE60846081"

robot1 = SO100Follower(SO100FollowerConfig(port=PORT1, id="arm1", use_degrees=True))
robot2 = SO100Follower(SO100FollowerConfig(port=PORT2, id="arm2", use_degrees=True))

robot1.connect()
robot2.connect()

names1 = list(robot1.bus.motors)
names2 = list(robot2.bus.motors)

print("arm1 motors:", names1)
print("arm2 motors:", names2)

for m in names1:
    robot1.bus.write("Torque_Enable", m, 0)

for m in names2:
    robot2.bus.write("Torque_Enable", m, 0)

robot1.disconnect()
robot2.disconnect()

print("Torque disabled on both arms.")