from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math


EPSILON = 5e-2


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)

    # if the robot is already holding a package, then return credit + reward - cost
    if robot.package is not None:
        reward = manhattan_distance(robot.package.position, robot.package.destination)
        cost = manhattan_distance(robot.position, robot.package.destination)
        return robot.credit + reward - cost

    # if the robot isn't holding a package, then take the closest avalibale package to it,
    avail = [p for p in env.packages if p.on_board]
    p = sorted(avail, key=lambda p: manhattan_distance(robot.position, p.position))[0]

    # and then return credit - position to the closest package to the robot
    cost = manhattan_distance(robot.position, p.position)
    return robot.credit - cost


class AgentGreedyImproved(AgentGreedy):
    def h(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


def Utility(env: WarehouseEnv, robot_id: int):
    if env.done():
        robot = env.get_robot(robot_id)
        other_robot = env.get_robot((robot_id + 1) % 2)
        if robot.credit > other_robot.credit:
            return 1
        elif robot.credit < other_robot.credit:
            return -1
        else:
            return 0
    return None


class RBAgent(Agent):
    def __init__(self):
        self.start_time: float = 0
        self.time_limit: float = 0

    def check_time(self):
        if time.time() - self.start_time >= self.time_limit - EPSILON:
            raise TimeoutError

    def h(self, env: WarehouseEnv, robot_id: int):
        if env.done():
            return Utility(env, robot_id)
        return smart_heuristic(env, robot_id) - smart_heuristic(env, (robot_id + 1) % 2)

    def generate_children(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return children, operators

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        depth = 0
        best_move = "park"
        self.start_time = time.time()
        self.time_limit = time_limit
        curr_max = -math.inf

        while True:
            try:

                children, operators = self.generate_children(env, robot_id)

                for child, op in zip(children, operators):
                    value = self.search(child, robot_id, depth, turn=(robot_id + 1) % 2)
                    if value > curr_max:
                        curr_max = value
                        best_move = op

                depth += 1

            except TimeoutError:
                return best_move


class AgentMinimax(RBAgent):
    # TODO: section b : 1

    def search(self, env: WarehouseEnv, robot_id: int, depth: int, turn: int):
        return self.minimax(env, robot_id, depth, turn)

    def minimax(self, env: WarehouseEnv, robot_id: int, depth: int, turn: int):
        self.check_time()

        if env.done() or depth == 0:
            return self.h(env, robot_id)

        children, _ = self.generate_children(env, turn)

        if turn == robot_id:
            curr_max = -math.inf
            for child in children:
                value = self.minimax(child, robot_id, depth - 1, (turn + 1) % 2)
                curr_max = min(value, curr_max)
            return curr_max

        else:  # turn == other_robot_id
            curr_min = math.inf
            for child in children:
                new_min = self.minimax(child, robot_id, depth - 1, (turn + 1) % 2)
                curr_min = min(new_min, curr_min)
            return curr_min


class AgentAlphaBeta(RBAgent):
    # TODO: section c : 1
    def search(self, env: WarehouseEnv, robot_id: int, depth: int, turn: int):
        return self.alphabeta(env, robot_id, depth, turn, -math.inf, math.inf)

    def alphabeta(
        self, env: WarehouseEnv, robot_id: int, depth: int, turn: int, alpha, beta
    ):
        self.check_time()

        if env.done() or depth == 0:
            return self.h(env, robot_id)

        children, _ = self.generate_children(env, robot_id)

        if turn == robot_id:
            curr_max = -math.inf
            for child in children:
                value = self.alphabeta(
                    child, robot_id, depth - 1, (turn + 1) % 2, alpha, beta
                )
                curr_max = max(value, curr_max)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return math.inf
            return curr_max

        else:  # turn == other_robot_id
            curr_min = math.inf
            for child in children:
                new_min = self.alphabeta(
                    child, robot_id, depth - 1, (turn + 1) % 2, alpha, beta
                )
                curr_min = min(new_min, curr_min)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return -math.inf
            return curr_min


class AgentExpectimax(RBAgent):
    # TODO: section d : 1
    def search(self, env: WarehouseEnv, robot_id: int, depth: int, turn: int):
        return self.expectimax(env, robot_id, depth, turn)

    def expectimax(self, env: WarehouseEnv, robot_id: int, depth: int, turn: int):
        self.check_time()

        if env.done() or depth == 0:
            return self.h(env, robot_id)

        children, operators = self.generate_children(env, robot_id)

        if turn == robot_id:
            curr_max = -math.inf
            for child in children:
                value = self.expectimax(child, robot_id, depth - 1, (turn + 1) % 2)
                curr_max = max(value, curr_max)
            return curr_max

        else:  # turn == other_robot_id
            probability = [1] * len(operators)

            for i, (child, op) in enumerate(zip(children, operators)):
                if op == "pick up" or op == "move east":
                    probability[i] *= 2

            total = sum(probability)
            probability = [p / total for p in probability]

            expctance = 0
            for child, p in zip(children, probability):
                expctance += p * self.expectimax(child, robot_id, depth - 1, (turn + 1) % 2)
            
            return expctance


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = [
            "move north",
            "move east",
            "move north",
            "move north",
            "pick_up",
            "move east",
            "move east",
            "move south",
            "move south",
            "move south",
            "move south",
            "drop_off",
        ]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
