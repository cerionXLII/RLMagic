# This class defines the schedule environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np

#Create a class for the schedule problem using the gym interface
class ScheduleGym(gym.Env):
    def __init__(self, num_days, num_hours, num_classes, num_subjects, verbose=False, render_mode=None):
        self.num_days = num_days
        self.num_hours = num_hours
        self.num_classes = num_classes
        self.num_subjects = num_subjects
        self.num_slots = num_days * num_hours
        self.target_hours = np.zeros((num_classes, num_subjects), dtype=int) # Target hours for each class and subject
        self.schedule = -1*np.ones((num_classes, num_days, num_hours), dtype=int) # Schedule for each class, -1 means no subject assigned
        self.num_actions_left = 1000 #Number of actions left to take
        self.verbose = verbose #Debug on or off
        self.initial_hours_to_assign = 0 #How many hours we have initially to assign, used to calculate score later
        self.render_mode = render_mode
        self.action_space = spaces.MultiDiscrete(self.get_action_sizes())
        self.observation_space = spaces.Box(low=0, high=1, shape=self.get_state_sizes(), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
        for class_id in range(self.num_classes):
            for subject_id in range(self.num_subjects):
                self.target_hours[class_id, subject_id] = np.random.randint(1, 5)
        
        self.initial_hours_to_assign = self.target_hours.sum().sum()

        self.schedule = -1*np.ones((self.num_classes, self.num_days, self.num_hours), dtype=int) # Schedule for each class, -1 means no subject assigned
        self.num_actions_left = self.initial_hours_to_assign * 10 #Optimally we would need to take number of hours to assign steps to complete the schedule, lets give it some wiggle room
        
        #Return the current state,info
        info = {}

        return self.state2vector(), info
    
    def state2vector(self):
        #Convert the state to a vector
        return np.concatenate([self.target_hours.flatten(), self.schedule.flatten()])/(self.initial_hours_to_assign * 10.0 + 1.0)
    
    def render(self):
        #Print the schedule
        for class_id in range(self.num_classes):
            print(f"Class {class_id + 1}:")
            for day in range(self.num_days):
                print(f"Day {day + 1}: {self.schedule[class_id, day]}")
            print()
        print(f'Fitness: {self.fitness()}, Actions left: {self.num_actions_left}')

        #print the target hours
        print("Target Hours:")
        for class_id in range(self.num_classes):
            print(f"Class {class_id + 1}: {self.target_hours[class_id]}")

    def step(self, action):
        # Update the schedule based on the action

        add_action = (action[0] == 1) # 1 is add, 0 is remove
        class_id = action[1]
        day = action[2]
        hour= action[3]
        subject_id = action[4]
         
        
        current_subject_id = self.schedule[class_id, day, hour]
        

        pre_action_fitness = self.fitness()
        reward = -1.1 #Cost of performing an action

        if self.verbose:
            print('Before action:')
            for class_id in range(self.num_classes):
                print(f"Class {class_id + 1}: {self.target_hours[class_id]}")
 
        #The slot was already booked, lets reomove it (and all of its dependencies)
        result = "N/A"
        
        #Lets see if it is an add action and if it is valid
        if (add_action and
            current_subject_id == -1 and #Slot is empty
            subject_id < self.num_subjects and #Not out of bounds
            self.target_hours[class_id, subject_id] > 0): #We have hours to assign

            #All valid, lets add it
            self.schedule[class_id, day, hour] = subject_id
            self.target_hours[class_id, subject_id] -= 1
            result = "Added"

        elif current_subject_id != -1:
            #It is a remove action, and there is a subject in the slot
            self.schedule[class_id, day, hour] = -1
            self.target_hours[class_id, current_subject_id] += 1
            result = "Removed"



        self.num_actions_left -= 1

        #Calculate the delta fitness
        after_action_fitness = self.fitness()
        reward +=  after_action_fitness - pre_action_fitness
        done = self.is_done()  
        if done:
            reward += self.initial_hours_to_assign * 10.0 #Completion bonus
            reward += after_action_fitness #If we are done we get the full score of the schedule
        
        if self.verbose:
            print(f"Action: {action}, Class: {class_id}, Subject: {subject_id}, Day: {day}, Hour: {hour}, , current_subject_id: {current_subject_id}, Result: {result}, reward: {reward}, done: {done}")     
            print('After action:')
            for class_id in range(self.num_classes):
                print(f"Class {class_id + 1}: {self.target_hours[class_id]}")

            if done:
                print('Final schedule:')
                self.render()
                
        truncated, info = False, {} #To be implemented later if needed
        #Return the next state, reward, done, truncated and info
        #return (self.target_hours, self.schedule), reward, done, truncated, info
        return self.state2vector(), reward, done, truncated, info       


    def is_done(self):
        #Check if the schedule is complete
        return np.all(self.target_hours == 0) or self.num_actions_left <= 0
    
    
    def get_action_sizes(self):
        #Return the sizes of the action space
        #return [self.num_classes, self.num_days, self.num_hours, self.num_subjects]
        #First action is add/remove action. If it is 1, we add a subject if it is 0 we remove a subject
        return [2, self.num_classes, self.num_days, self.num_hours, self.num_subjects] 
        
    
    def get_state_sizes(self):
        #Return the shapes of the state space
        #return self.target_hours.shape, self.schedule.shape
        return self.state2vector().shape

    
    def fitness(self):
        #Calculate the fitness of the schedule
        #fitness = self.num_actions_left * 0.001 #We want to maximize the number of actions left
        fitness = 0.0
        # target hours remaining
        target_hours_remaining = self.target_hours.sum().sum()

        fitness -= target_hours_remaining * 1

        # Count the number of holes in the schedule, that is where no subject is assigned, but is surrounded by subjects
        # If there are no subjects assigned to the edges, that is not considered a hole
        num_holes = 0

        # Create shifted versions of the schedule to compare adjacent hours
        left_shifted = np.roll(self.schedule, shift=-1, axis=2)
        right_shifted = np.roll(self.schedule, shift=1, axis=2)

        # Identify holes: -1 in the current schedule, and not -1 in both the left and right shifted schedules
        # Avoid considering the edges by setting the comparison for the first and last hour to False
        holes = (self.schedule == -1) & (left_shifted != -1) & (right_shifted != -1)
        holes[:, :, 0] = False  # Ignore first hour edge cases
        holes[:, :, -1] = False  # Ignore last hour edge cases

        # Count the number of holes
        num_holes = np.sum(holes)

        fitness -= num_holes * 0.15

        return fitness