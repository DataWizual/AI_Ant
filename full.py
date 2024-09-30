import matplotlib.pyplot as plt
from IPython import display
from copy import deepcopy
from collections import deque
import torch.nn as nn
import torch
import math
import pygame
import random
import numpy as np
import random

WIDTH = 800
HEIGHT = 800
ANTS = 5
FOOD = 20
FPS = 30
IMAGE_W = 40
IMAGE_H = 30
NEST_X, NEST_Y = 100, 100
DISTANCE = 15
FOOD_DIST = 50
P_WHITE = (186, 185, 179)


class OUNoise:
    def __init__(self, action_dimention, mu=0, theta=0.15, sigma=0.3):
        self.action_dimention = action_dimention
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimention) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimention) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x+dx
        return self.state


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim, output_tanh):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.layer3 = nn.Linear(layer2_dim, output_dim)
        self.output_tanh = output_tanh
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        hidden = self.layer1(input)
        hidden = self.leaky_relu(hidden)
        hidden = self.layer2(hidden)
        hidden = self.leaky_relu(hidden)
        output = self.layer3(hidden)
        if self.output_tanh:
            return self.tanh(output)
        else:
            return output


class DDPG():
    def __init__(self, state_dim, action_dim, action_scale, noise_decrease,
                 gamma=0.999, batch_size=64, q_lr=1e-4, pi_lr=1e-5, tau=0.001, memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.pi_model = NeuralNetwork(
            self.state_dim, 400, 300, self.action_dim, output_tanh=True).to(self.device)
        self.q_model = NeuralNetwork(
            self.state_dim + self.action_dim, 400, 300, 1, output_tanh=False).to(self.device)
        self.pi_target_model = deepcopy(self.pi_model).to(self.device)
        self.q_target_model = deepcopy(self.q_model).to(self.device)
        self.noise = OUNoise(self.action_dim)
        self.noise_threshold = 1
        self.noise_decrease = noise_decrease
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=q_lr)
        self.pi_optimizer = torch.optim.Adam(
            self.pi_model.parameters(), lr=pi_lr)
        self.memory = deque(maxlen=memory_size)

    def get_action(self, state):
        # Convert state to tensor and move to device
        state_tensor = torch.FloatTensor(state).to(self.device)
        pred_action = self.pi_model(state_tensor).detach().cpu().numpy()
        action = pred_action + self.noise_threshold * self.noise.sample()
        return action

    def fit(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))
        if len(self.memory) >= self.batch_size:
            # Sample a batch of transitions
            batch = random.sample(self.memory, self.batch_size)
            # Convert batch to numpy arrays for efficient tensor creation
            states, actions, rewards, dones, next_states = map(
                np.array, zip(*batch))
            # Convert numpy arrays to tensors and move to the appropriate device
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(
                actions, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).reshape(
                self.batch_size, 1).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).reshape(
                self.batch_size, 1).to(self.device)
            next_states = torch.tensor(
                next_states, dtype=torch.float32).to(self.device)
            # Update Q-Model
            pred_next_actions = self.pi_target_model(next_states)
            next_states_and_pred_next_actions = torch.cat(
                (next_states, pred_next_actions), dim=1)
            targets = rewards + self.gamma * \
                (1 - dones) * self.q_target_model(next_states_and_pred_next_actions)
            actions = actions.unsqueeze(1)
            states_and_actions = torch.cat((states, actions), dim=1)
            q_loss = torch.mean(
                (targets.detach() - self.q_model(states_and_actions)) ** 2)
            self.update_target_model(
                self.q_target_model, self.q_model, self.q_optimizer, q_loss)
            pred_actions = self.pi_model(states)
            states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
            pi_loss = -torch.mean(self.q_model(states_and_pred_actions))
            self.update_target_model(self.pi_target_model,
                                     self.pi_model, self.pi_optimizer, pi_loss)
        if self.noise_threshold > 0:
            self.noise_threshold = max(
                0, self.noise_threshold - self.noise_decrease)

    def update_target_model(self, target_model, model, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=0.5)
        optimizer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data)

    def save_model(self, filepath="ddpg_checkpoint.pth"): 
        """Saving models, optimizers, and agent parameters."""  # Translated comment
        checkpoint = { 
            'pi_model_state_dict': self.pi_model.state_dict(),  # Weights of the pi model
            'q_model_state_dict': self.q_model.state_dict(),    # Weights of the q model
            'pi_target_model_state_dict': self.pi_target_model.state_dict(),  # Weights of the target pi model
            'q_target_model_state_dict': self.q_target_model.state_dict(),    # Weights of the target q model
            'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),  # pi optimizer
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),    # q optimizer
            'noise_state': self.noise.state,  # Noise state
            'noise_threshold': self.noise_threshold,  # Noise parameter
            'tau': self.tau,  # Parameter τ for updating the target model
            'gamma': self.gamma,  # Discount factor
            'memory': self.memory,  # Agent memory (if needed to be saved)
        } 
        torch.save(checkpoint, filepath) 
        print(f"Model saved at {filepath}")  # Translated print statement

    def load_model(self, filepath="ddpg_checkpoint.pth"): 
        """Loading models, optimizers, and agent parameters."""  # Translated comment
        checkpoint = torch.load(filepath, map_location=self.device) 
        # Loading model states
        self.pi_model.load_state_dict(checkpoint['pi_model_state_dict']) 
        self.q_model.load_state_dict(checkpoint['q_model_state_dict']) 
        self.pi_target_model.load_state_dict(checkpoint['pi_target_model_state_dict'])  # Target pi model
        self.q_target_model.load_state_dict(checkpoint['q_target_model_state_dict'])    # Target q model
        # Loading optimizer states
        self.pi_optimizer.load_state_dict(checkpoint['pi_optimizer_state_dict']) 
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict']) 
        # Loading noise state
        self.noise.state = checkpoint.get('noise_state', np.ones(self.action_dim))  # Restoring noise state
        # Loading other parameters
        self.noise_threshold = checkpoint.get('noise_threshold', 1) 
        self.tau = checkpoint.get('tau', 1e-3) 
        self.gamma = checkpoint.get('gamma', 0.99) 
        # If memory needs to be restored
        self.memory = checkpoint.get('memory', deque(maxlen=100000)) 
        print(f"Model loaded from {filepath}")  # Translated print statement



class Ant(pygame.sprite.Sprite):
    def __init__(self, delta_time=1, setup_instance=None, agent=None):
        super().__init__()
        self.delta_time = delta_time
        self.speed = 0.1 * delta_time
        self.setup_instance = setup_instance
        self.agent = agent
        self.is_demo_mode = True
        self.demo_steps = 500
        self.steps = 0
        self.index = 0
        self.size = (IMAGE_W, IMAGE_H)
        self.image = pygame.transform.smoothscale(pygame.image.load(
            "stand.png").convert_alpha(), self.size)
        self.move = [pygame.transform.smoothscale(pygame.image.load(
            f'ant{i}.png').convert_alpha(), self.size) for i in range(1, 17)]
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.x, self.y = NEST_X, NEST_Y
        self.angle = random.uniform(0, 2*np.pi)
        self.encountered_foods = set()
        self.dragged_food = None
        self.choose = 0
        self.prev_state = None
        self.score = 0
        self.steps = 0

    def update(self, screen, nest_rect, nest_mask, foods):
        self.movement()
        state = self.get_state(foods, nest_rect)
        angle_change = self.agent.get_action(state)
        if isinstance(angle_change, np.ndarray) and angle_change.size == 1:
            angle_change = angle_change.item()
        if self.choose == 0:
            self.turn_to_food(foods, angle_change)
        elif self.choose == 1:
            self.turn_to_nest(nest_rect, nest_mask, angle_change)
        self.rotate_center(screen)
        self.wrap_around()
        next_state = self.get_state(foods, nest_rect)
        reward = self.get_reward(foods, nest_rect, angle_change)
        done = self.is_collision()
        if done:
            self.score = 0
            self.setup_instance.create_objects()
        if self.agent and self.prev_state:
            self.agent.fit(self.prev_state, angle_change,
                           reward, done, next_state)
        self.prev_state = next_state
        # self.steps += 1
        # if self.steps % 1000 == 0:
        #     self.agent.save_model(
        #         filepath="ddpg_checkpoint.pth")

    def movement(self):
        self.speedx = self.speed * math.cos(self.angle)
        self.speedy = -self.speed * math.sin(self.angle)
        self.x += self.speedx
        self.y += self.speedy
        self.image = self.move[self.index]
        self.index = (self.index + 1) % 16

    def wrap_around(self):
        if self.x < DISTANCE:
            angle_d = self.angle_direction(-0.25, 0.25)
            self.angle += angle_d * 0.1
        elif self.x >= WIDTH-DISTANCE:
            angle_d = self.angle_direction(0.75, 1.25)
            self.angle += angle_d * 0.1
        elif self.y < DISTANCE:
            angle_d = self.angle_direction(1.25, 1.75)
            self.angle += angle_d * 0.1
        elif self.y >= HEIGHT-DISTANCE:
            angle_d = self.angle_direction(0.25, 0.75)
            self.angle += angle_d * 0.1

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def clamp_position(self):
        self.x = max(0, min(self.x, WIDTH))
        self.y = max(0, min(self.y, HEIGHT))

    def get_state(self, foods, nest_rect):
        self.clamp_position()
        normalized_x = self.x / WIDTH
        normalized_y = self.y / HEIGHT
        normalized_angle = (self.normalize_angle(
            self.angle) + np.pi) / (2 * np.pi)
        if self.choose == 0:
            angle_d, _ = self.find_food(foods)
            angle_different = (angle_d + np.pi) / (2 * np.pi)
            state = [normalized_x, normalized_y,
                     normalized_angle,  angle_different]
            return state
        elif self.choose == 1:
            angle_d, _ = self.find_nest(nest_rect)
            angle_different = (angle_d + np.pi) / (2 * np.pi)
            state = [normalized_x, normalized_y,
                     normalized_angle,  angle_different]
            return state

    def get_reward(self, foods, nest_rect, angle_change):
        reward = 0
        max_dist = np.linalg.norm([WIDTH, HEIGHT])
        if self.choose == 0:
            angle_d, min_distance = self.find_food(foods)
            _, _, distance = min_distance
            reward += 5*math.cos(angle_d) + 1
            reward += 6*(1 - min(distance / max_dist, 1))
            if self.is_collision():
                reward -= 5
            self.previous_distance = distance
            self.previous_angle_d = angle_d
            return reward
        elif self.choose == 1:
            angle_d, min_distance = self.find_nest(nest_rect)
            reward += 5*math.cos(angle_d) + 1
            reward += 6*(1 - min(min_distance / max_dist, 1))
            if self.is_collision():
                reward -= 5
            self.previous_distance = min_distance
            self.previous_angle_d = angle_d
            return reward

    def is_collision(self):
        return (self.x < DISTANCE or self.x >= WIDTH-DISTANCE or
                self.y < DISTANCE or self.y >= HEIGHT-DISTANCE)

    def find_food(self, foods):
        dist_list = []
        for food in foods:
            if not food.taken:
                points = (self.x, self.y)
                goal = [(food.x, food.y)]
                dist_to = self.get_closest_point(points, goal)
                dist_list.append((food.x, food.y, dist_to))
        if dist_list:
            angle_f, min_distance = self.min_dist(dist_list)
            angle_d = self.angle_different(angle_f)
            return angle_d, min_distance

    def turn_to_food(self, foods, angle_d):
        self.angle += angle_d * 0.25
        for food in foods:
            offset = (int(food.rect.left - self.rect.left),
                      int(food.rect.top - self.rect.top))
            if self.mask.overlap(food.mask, offset):
                if food not in self.encountered_foods and not food.taken:
                    self.dragged_food = food
                    self.encountered_foods.add(self.dragged_food)
                    self.angle += np.pi * 0.25
                    food.taken = True
                    self.choose = 1
                    self.score += 1
                    break

    def find_nest(self, nest_rect):
        points = (self.x, self.y)
        goal = (NEST_X, NEST_Y)
        dist_to = self.get_closest_point(points, goal)
        angle_f = self.arctan_calc(nest_rect.centerx, nest_rect.centery)
        angle_d = self.angle_different(angle_f)
        return angle_d, dist_to

    def turn_to_nest(self, nest_rect, nest_mask, angle_d):
        self.angle += angle_d * 0.25
        offset = (int(nest_rect.left - self.rect.left),
                  int(nest_rect.top - self.rect.top))
        if self.mask.overlap(nest_mask, offset):
            if self.dragged_food is not None:
                self.dragged_food.kill()
                self.dragged_food = None
                self.angle += np.pi * 0.25
                if self.setup_instance is not None:
                    self.setup_instance.spawn_food()
                self.choose = 0

    def angle_direction(self, start, end):
        angle_dir = random.uniform(start * np.pi, end * np.pi)
        angle_direct = self.angle_different(angle_dir)
        return angle_direct

    def angle_different(self, direction):
        angle_diff = (direction - self.angle +
                      np.pi) % (2 * np.pi) - np.pi
        return angle_diff

    def get_closest_point(self, points, goal):
        p1 = np.array(points)
        p2 = np.array(goal)
        dist_to = np.linalg.norm(p1 - p2)
        return dist_to

    def min_dist(self, dist):
        min_distance = min(dist, key=lambda x: x[2])
        min_p_x, min_p_y, _ = min_distance
        angle_f = self.arctan_calc(min_p_x, min_p_y)
        return angle_f, min_distance

    def arctan_calc(self, x, y):
        angle = np.arctan2(self.y - y, x - self.x)
        return angle

    def rotate_center(self, screen):
        angle_in_degrees = np.degrees(self.angle)
        rotate_image = pygame.transform.rotate(self.image, angle_in_degrees)
        self.rect = rotate_image.get_rect(center=self.rect.center)
        screen.blit(rotate_image, self.rect)

    def draw(self):
        self.rect.center = (self.x, self.y)

    def ant_coord(self):
        self.food_center_x = self.x + DISTANCE * \
            np.cos(self.angle)
        self.food_center_y = self.y - DISTANCE * \
            np.sin(self.angle)
        return self.food_center_x, self.food_center_y


class Food(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.image = pygame.transform.smoothscale(pygame.image.load(
            "corn.png").convert_alpha(), (27, 16))
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect.center = (self.x, self.y)
        self.angle = random.uniform(0, 2 * np.pi)
        self.taken = False

    def draw(self, screen):
        self.rotate_image = pygame.transform.rotate(
            self.image, np.degrees(self.angle))
        screen.blit(self.rotate_image, self.rect)


class Setup:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Ant colony')
        icon = pygame.image.load('ant_ico.png')
        pygame.display.set_icon(icon)
        self.clock = pygame.time.Clock()
        self.delta_time = self.clock.tick(FPS)
        self.agent = DDPG(state_dim=4, action_dim=1,
                          action_scale=1, noise_decrease=0.0001)
        # self.agent.load_model(filepath="ddpg_checkpoint.pth")

        self.create_objects()

    def draw_nest(self):
        nest = pygame.transform.smoothscale(pygame.image.load(
            "nest.png").convert_alpha(), (270, 200))
        nest_rect = nest.get_rect(center=(NEST_X, NEST_Y))
        nest_mask = pygame.mask.from_surface(nest)
        self.screen.blit(nest, nest_rect)
        return nest_rect, nest_mask

    def create_objects(self):
        self.ants = pygame.sprite.Group(
            [Ant(self.delta_time, self, self.agent) for _ in range(ANTS)])
        self.foods = pygame.sprite.Group(
            [Food(random.uniform(FOOD_DIST, WIDTH-FOOD_DIST),
                  random.uniform(FOOD_DIST, HEIGHT-FOOD_DIST)) for _ in range(FOOD)])

    def type_of_ants(self, nest_rect, nest_mask):
        for ant in self.ants:
            ant.update(self.screen, nest_rect, nest_mask, self.foods)
            ant.draw()
            self.score = ant.score
            x, y = ant.ant_coord()
            if isinstance(x, np.ndarray) and x.size == 1:
                x = x.item()
            if isinstance(y, np.ndarray) and y.size == 1:
                y = y.item()
            if ant.dragged_food:
                ant.dragged_food.rect.center = (x, y)

    def spawn_food(self):
        new_food = Food(random.uniform(FOOD_DIST, WIDTH-FOOD_DIST),
                        random.uniform(FOOD_DIST, HEIGHT-FOOD_DIST))
        self.foods.add(new_food)

    def main(self):
        while True:
            self.screen.fill(P_WHITE)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.create_objects()
            nest_rect, nest_mask = self.draw_nest()
            for food in self.foods:
                food.draw(self.screen)
            self.type_of_ants(nest_rect, nest_mask)

            pygame.display.update()
            self.clock.tick(FPS)


if __name__ == '__main__':
    setup_instance = Setup()
    setup_instance.main()
