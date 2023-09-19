from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

environment = "SuperMarioBros-1-1-v0"

action_space = SIMPLE_MOVEMENT

transfer = True

lr = 1e-4

buffer_capacity = 20000

num_epochs = 200

batch_size = 32

render = True

initialLearning = 10000

targetUpdateFrequency = 1000

gamma = 0.99

start = 0.4
frames = 10000

epsFinal = 0.01
# chage the epsStart fot continuing training
epsStart = 0.1
decay = 100000
