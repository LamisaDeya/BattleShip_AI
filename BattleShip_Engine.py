# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:06:39 2024

@author: ASUS
"""


import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import random
import math


# Class representing a ship
class Ship:
    # Determine positions based on size and orientation

    def __init__(self, size, row=None, col=None, orientation=None):
        self.size = size
        self.row = row if row is not None else random.randint(0, 9)
        self.col = col if col is not None else random.randint(0, 9)
        self.orientation = orientation if orientation is not None else random.choice(["h", "v"])
        self.indexes = self.compute_indexes()

    def compute_indexes(self):
        indexes = []
        if self.orientation == "h":
            for i in range(self.size):
                indexes.append(self.row * 10 + self.col + i)
        else:
            for i in range(self.size):
                indexes.append((self.row + i) * 10 + self.col)
        return indexes
        
def is_valid_placement(ship, other_ships):
    for other in other_ships:
        if set(ship.indexes).intersection(set(other.indexes)):
            return False
    return True

# Class representing a player
class Player:
    def __init__(self, use_ga=True):
        self.ships = []  # List of ships
        self.search = ["U" for _ in range(100)]  # Search grid: unknown ('U'), hit ('H'), sunk ('S'), miss ('M')
        if use_ga:
            self.place_ships_with_ga()
        else:
            self.place_ships()
        self.update_indexes()  # Update flattened list of ship indexes

    def valid_placement(self, ship):
        for idx in ship.indexes:
            if idx < 0 or idx >= 100:
                return False
            row, col = divmod(idx, 10)
            if ship.orientation == 'h' and (col + ship.size - 1) >= 10:
                return False
            if ship.orientation == 'v' and (row + ship.size - 1) >= 10:
                return False
        for s in self.ships:
            if set(ship.indexes).intersection(s.indexes):
                return False
        return True

    def place_ships(self):
        ship_sizes = [5, 4, 3, 3, 2]
        for size in ship_sizes:
            placed = False
            while not placed:
                orientation = random.choice(["h", "v"])
                if orientation == "h":
                    row = random.randint(0, 9)
                    col = random.randint(0, 10 - size)
                else:
                    row = random.randint(0, 10 - size)
                    col = random.randint(0, 9)
                
                new_ship = Ship(size=size, row=row, col=col, orientation=orientation)
                if self.valid_placement(new_ship):
                    self.ships.append(new_ship)
                    placed = True
        self.update_indexes()  # Ensure indexes are updated after placement

    def place_ships_with_ga(self):
        board_size = 10  # assuming a 10x10 board
        ship_sizes = [5, 4, 3, 3, 2]  # standard ship sizes
        population_size = 30  # Increased population size for better GA performance
        generations = 150  # Increased generations for more thorough evolution
        mutation_rate = 0.05  # Reduced mutation rate for better convergence

        ga = GeneticAlgorithm(population_size, mutation_rate, generations, board_size)
        best_placement = ga.evolve_population(
            ga.initialize_population(population_size, ship_sizes, board_size),
            generations, mutation_rate, board_size
        )
        if self.valid_placement_for_all(best_placement):
            print("GA produced valid placement")
            self.ships = [
                Ship(size=size, row=row, col=col, orientation=orientation)
                for (row, col, orientation, size) in best_placement
            ]
        else:
            print("GA produced invalid placement, using random placement instead.")
            self.place_ships()  # Fallback to random placement if GA fails
        self.update_indexes()

    def valid_placement_for_all(self, placement):
        occupied_positions = set()
        for row, col, orientation, size in placement:
            ship = Ship(size=size, row=row, col=col, orientation=orientation)
            
            # Check for boundary violations
            if ship.orientation == 'h' and col + size > 10:
                print(f"Ship {ship} out of horizontal bounds.")
                return False
            if ship.orientation == 'v' and row + size > 10:
                print(f"Ship {ship} out of vertical bounds.")
                return False
            
            # Check for overlapping ships
            for idx in ship.indexes:
                if idx in occupied_positions:
                    print(f"Overlap detected for ship {ship} at index {idx}.")
                    return False
                occupied_positions.add(idx)
        return True
    

    def update_indexes(self):
        list_of_lists = [ship.indexes for ship in self.ships]
        self.indexes = [index for sublist in list_of_lists for index in sublist]

    def reset(self):
        self.ships = []
        self.search = ["U" for _ in range(100)]
        self.place_ships()
        self.update_indexes()

    def display_ship_placement(self):
        indexes = ["-" if i not in self.indexes else "X" for i in range(100)]
        for row in range(10):
            print(" ".join(indexes[(row) * 10:(row + 1) * 10]))

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, grid_size):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.grid_size = grid_size

    def initialize_population(self, pop_size, ship_sizes, board_size):
        population = []
        for _ in range(pop_size):
            placement = []
            for size in ship_sizes:
                valid = False
                while not valid:
                    orientation = random.choice(['h', 'v'])
                    if orientation == 'h':
                        row = random.randint(1, board_size - 2)
                        col = random.randint(1, board_size - size)
                    else:
                        row = random.randint(1, board_size - size)
                        col = random.randint(1, board_size - 2)
                    new_ship = (row, col, orientation, size)
                    if self.valid_ship_placement(placement, new_ship):
                        placement.append(new_ship)
                        valid = True
            population.append(placement)
        return population

    def valid_ship_placement(self, existing_placements, new_ship):
        # Ensure new_ship is a tuple of (row, col, orientation, size)
        if not isinstance(new_ship, tuple) or len(new_ship) != 4:
            #print(f"Invalid new_ship format: {new_ship}")
            return False

        row, col, orientation, size = new_ship
        ship_indexes = []
        if orientation == 'h':
            ship_indexes = [(row, col + k) for k in range(size)]
        elif orientation == 'v':
            ship_indexes = [(row + k, col) for k in range(size)]

        # Check if any index is out of bounds
        for r, c in ship_indexes:
            if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                #print(f"Ship out of bounds: {new_ship}")
                return False

        # Ensure existing_placements contains only tuples of (row, col, orientation, size)
        for existing_ship in existing_placements:
            if not isinstance(existing_ship, tuple) or len(existing_ship) != 4:
                #print(f"Invalid existing_ship format: {existing_ship}")
                continue  # Skip invalid entries

            er, ec, e_orientation, e_size = existing_ship
            existing_indexes = []
            if e_orientation == 'h':
                existing_indexes = [(er, ec + k) for k in range(e_size)]
            elif e_orientation == 'v':
                existing_indexes = [(er + k, ec) for k in range(e_size)]

            if set(ship_indexes).intersection(existing_indexes):
                #print(f"Overlap detected with ship: {new_ship} and existing ship: {existing_ship}")
                return False

        return True

    def fitness(self, placement):
        total_distance = 0
        edge_penalty = 0
        corner_penalty = 0
        coverage_bonus = 0

        occupied_positions = set()

        for i in range(len(placement)):
            row, col, orientation, size = placement[i]

            ship_indexes = []
            if orientation == 'h':
                ship_indexes = [(row, col + k) for k in range(size)]
            elif orientation == 'v':
                ship_indexes = [(row + k, col) for k in range(size)]

            occupied_positions.update(ship_indexes)

            for (r, c) in ship_indexes:
                if r == 0 or r == self.grid_size - 1 or c == 0 or c == self.grid_size - 1:
                    edge_penalty += 1
                if (r == 0 and c == 0) or (r == 0 and c == self.grid_size - 1) or (r == self.grid_size - 1 and c == 0) or (r == self.grid_size - 1 and c == self.grid_size - 1):
                    corner_penalty += 2

            for j in range(i + 1, len(placement)):
                other_row, other_col, other_orientation, other_size = placement[j]
                distance = abs(row - other_row) + abs(col - other_col)
                total_distance += distance

        unique_rows = {r for r, c in occupied_positions}
        unique_cols = {c for r, c in occupied_positions}
        coverage_bonus = len(unique_rows) + len(unique_cols)

        fitness_score = total_distance + coverage_bonus - (edge_penalty + corner_penalty)
        return fitness_score

    def select_parents(self, population, fitnesses):
        selected = random.choices(population, weights=fitnesses, k=2)
        return selected[0], selected[1]

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        if not self.valid_placement_for_all(child1):
            print(f"Invalid child1 from crossover: {child1}")
        if not self.valid_placement_for_all(child2):
            print(f"Invalid child2 from crossover: {child2}")
        
        return child1, child2

    def mutate(self, individual, mutation_rate, board_size):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                valid = False
                while not valid:
                    orientation = random.choice(['h', 'v'])
                    if orientation == 'h':
                        row = random.randint(0, board_size - 1)
                        col = random.randint(0, board_size - individual[i][3])
                    else:
                        row = random.randint(0, board_size - individual[i][3])
                        col = random.randint(0, board_size - 1)
                    new_ship = (row, col, orientation, individual[i][3])
                    if self.valid_ship_placement(individual[:i] + individual[i+1:], new_ship):
                        individual[i] = new_ship
                        valid = True
        return individual

    def evolve_population(self, population, generations, mutation_rate, board_size):
        for generation in range(generations):
            fitnesses = [self.fitness(ind) for ind in population]
            next_population = []
            for _ in range(len(population) // 2):
                parent1, parent2 = self.select_parents(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1, mutation_rate, board_size))
                next_population.append(self.mutate(child2, mutation_rate, board_size))
            population = next_population
            
            for individual in population:
                if any(len(ship) != 4 for ship in individual):
                    print("Found an individual with incorrect structure:", individual)

        fitnesses = [self.fitness(ind) for ind in population]
        best_index = fitnesses.index(max(fitnesses))
        best_placement = population[best_index]

        if not self.valid_placement_for_all(best_placement):
            print("Best placement after GA is invalid:", best_placement)
        else:
            print("Best placement after GA is valid:", best_placement)
        
        return best_placement

    def valid_placement_for_all(self, placement):
        occupied_positions = set()
        for row, col, orientation, size in placement:
            ship = Ship(size=size, row=row, col=col, orientation=orientation)
            if not self.valid_ship_placement(occupied_positions, (row, col, orientation, size)):
                print(f"Invalid placement detected for ship {ship}")
                return False
            occupied_positions.update(ship.indexes)
        return True

class Game:
    def __init__(self, human1, human2, player1, player2):
        self.human1 = human1
        self.human2 = human2
        self.player1 = player1
        self.player2 = player2
        self.player1_turn = True
        self.computer_turn = not self.human1
        self.over = False
        self.result = None
        self.n_shots = 0
        self.in_sinking_mode = False
        self.hit_stack = [] 
        self.actual_misses = set()  # Track actual miss cells
    def place_ship(self, ship, player):
        if is_valid_placement(ship, player.ships):
            player.ships.append(ship)
            player.update_indexes()
            return True
        return False
        
    def make_move(self, i):
        player = self.player1 if self.player1_turn else self.player2
        opponent = self.player2 if self.player1_turn else self.player1
        hit = False
        
        for ship in opponent.ships:
            if i in ship.indexes:
                player.search[i] = "H"
                hit = True
                sunk = True
                for idx in ship.indexes:
                    if player.search[idx] == "U":
                        sunk = False
                        break
                if sunk:
                    for idx in ship.indexes:
                        player.search[idx] = "S"
                    self.in_sinking_mode = False
                break

        if not hit:
            player.search[i] = "M"
            self.actual_misses.add(i)  # Track the actual miss
        
        game_over = True
        for ship in opponent.ships:
            if any(player.search[idx] == "U" for idx in ship.indexes):
                game_over = False
                break
        self.over = game_over
        if self.over:
            self.result = 1 if self.player1_turn else 2
            
        if not hit:
            self.player1_turn = not self.player1_turn
            if (self.human1 and not self.human2) or (not self.human1 and self.human2):
                self.computer_turn = not self.computer_turn
        self.n_shots += 1
        
        if not self.player1_turn:
            #print(player.search[i])
            if player.search[i]=='H':
                return 1 
            elif player.search[i]=='S':
                return 2

    def random_ai(self):
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U" and i not in self.actual_misses]
        if unknown:
            random_index = random.choice(unknown)
            num=self.make_move(random_index)
            return num

    def basic_ai(self):
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]
        
        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)
                
        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num=self.make_move(u)
                return num
                
        if unknown_with_neighbouring_hits1:
            num=self.make_move(random.choice(unknown_with_neighbouring_hits1))
            return num
            
        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num=self.make_move(random.choice(checker_board))
            return num
        
        self.random_ai()  
        
    def fuzzy_search(self, search_grid):
        # Define fuzzy variables
        hits = ctrl.Antecedent(np.arange(0, 5, 1), 'hits')
        unknowns = ctrl.Antecedent(np.arange(0, 5, 1), 'unknowns')
        probability = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'probability')

        # Membership functions
        hits['low'] = fuzz.trimf(hits.universe, [0, 0, 2])
        hits['medium'] = fuzz.trimf(hits.universe, [1, 2, 3])
        hits['high'] = fuzz.trimf(hits.universe, [2, 4, 4])

        unknowns['low'] = fuzz.trimf(unknowns.universe, [0, 0, 2])
        unknowns['medium'] = fuzz.trimf(unknowns.universe, [1, 2, 3])
        unknowns['high'] = fuzz.trimf(unknowns.universe, [2, 4, 4])

        probability['low'] = fuzz.trimf(probability.universe, [0, 0, 0.5])
        probability['medium'] = fuzz.trimf(probability.universe, [0.3, 0.5, 0.7])
        probability['high'] = fuzz.trimf(probability.universe, [0.5, 1, 1])

        # Define rules
        '''
        rule1 = ctrl.Rule(hits['low'] & unknowns['high'], probability['high'])
        rule2 = ctrl.Rule(hits['low'] & unknowns['medium'], probability['medium'])
        rule3 = ctrl.Rule(hits['low'] & unknowns['low'], probability['low'])
        rule4 = ctrl.Rule(hits['medium'] & unknowns['high'], probability['medium'])
        rule5 = ctrl.Rule(hits['medium'] & unknowns['medium'], probability['low'])
        rule6 = ctrl.Rule(hits['medium'] & unknowns['low'], probability['low'])
        rule7 = ctrl.Rule(hits['high'] & unknowns['high'], probability['low'])
        rule8 = ctrl.Rule(hits['high'] & unknowns['medium'], probability['low'])
        rule9 = ctrl.Rule(hits['high'] & unknowns['low'], probability['low'])
        '''
        
        
        # Define rules
        rule1 = ctrl.Rule(hits['low'] & unknowns['high'], probability['medium'])  # Exploratory, moderate chance
        rule2 = ctrl.Rule(hits['low'] & unknowns['medium'], probability['low'])   # Less likely, less unknowns
        rule3 = ctrl.Rule(hits['low'] & unknowns['low'], probability['low'])      # Unlikely, few unknowns and hits

        rule4 = ctrl.Rule(hits['medium'] & unknowns['high'], probability['high'])  # Good chance, could be part of a ship
        rule5 = ctrl.Rule(hits['medium'] & unknowns['medium'], probability['medium']) # Possible, but less unknowns
        rule6 = ctrl.Rule(hits['medium'] & unknowns['low'], probability['low'])    # Unlikely, explored area

        rule7 = ctrl.Rule(hits['high'] & unknowns['high'], probability['high'])    # Very likely, possible ship continuation
        rule8 = ctrl.Rule(hits['high'] & unknowns['medium'], probability['high'])  # Likely, possible ship
        rule9 = ctrl.Rule(hits['high'] & unknowns['low'], probability['medium'])   # Lower chance, but still possible

        # Control system creation
        probability_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        probability_simulation = ctrl.ControlSystemSimulation(probability_ctrl)

        max_probability = 0
        best_index = None

        # Evaluate each unknown cell
        for i, state in enumerate(search_grid):
            if state == 'U':
                neighboring_hits = 0
                neighboring_unknowns = 0

                for offset in [-1, 1, -10, 10]:  # Check immediate neighbors
                    neighbor_index = i + offset
                    if 0 <= neighbor_index < len(search_grid):
                        if search_grid[neighbor_index] == 'H':
                            neighboring_hits += 1
                        elif search_grid[neighbor_index] == 'U':
                            neighboring_unknowns += 1

                probability_simulation.input['hits'] = neighboring_hits
                probability_simulation.input['unknowns'] = neighboring_unknowns
                probability_simulation.compute()

                prob = probability_simulation.output['probability']
                if prob > max_probability:
                    max_probability = prob
                    best_index = i

        return best_index

    def basic_ai_with_fuzzy(self):
        search = self.player1.search if self.player1_turn else self.player2.search
        hits =[i for i, square in enumerate(search) if square == "H"]

        # If there are no hits yet, use fuzzy logic to find the first target
        if not hits:
            index = self.fuzzy_search(search)
            if index is not None:
                num=self.make_move(index)
                return num
        
        # If there are hits, follow the basic AI logic
        unknown =[i for i, square in enumerate(search) if square == "U"]
        
        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u+1 in hits or u-1 in hits or u+10 in hits or u-10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u+2 in hits or u-2 in hits or u+20 in hits or u-20 in hits:
                unknown_with_neighbouring_hits2.append(u)
                
        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
               num = self.make_move(u)
               return num
                
        if len(unknown_with_neighbouring_hits1) > 0:
            num=self.make_move(random.choice(unknown_with_neighbouring_hits1))
            return num
            
        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0: 
                checker_board.append(u)
        if len(checker_board) > 0:
            num=self.make_move(random.choice(checker_board))
            return num
        
        self.random_ai()
        
    '''
    def basic_ai_MM0(self):
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]

        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)
                    
        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num=self.make_move(u)
                return num
                    
        if unknown_with_neighbouring_hits1:
            num=self.make_move(random.choice(unknown_with_neighbouring_hits1))
            return num
                
        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num=self.make_move(random.choice(checker_board))
            return num

        self.in_sinking_mode = False
        self.minmax_ai()  # Switch to MinMax AI instead of random AI
      
    def minmax_ai0(self, depth=2):
        if self.in_sinking_mode:
            #self.basic_ai_MM()
            self.basic_ai_MM()
            return
        
        best_score = -math.inf
        best_move = None
        for move in self.get_available_moves():
            score = self.simulate_and_evaluate(move, depth, False, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_move = move
        if best_move is not None:
            num=self.make_move(best_move)
            
            if self.player1_turn:
                if self.player1.search[best_move] == "H":
                    self.in_sinking_mode = True
                    self.hit_stack.append(best_move)
            return num
        else:
            return(self.basic_ai())
        
    def basic_ai_MM1(self):
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]

        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)

        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num = self.make_move(u)
                return num

        if unknown_with_neighbouring_hits1:
            num = self.make_move(random.choice(unknown_with_neighbouring_hits1))
            return num

        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num = self.make_move(random.choice(checker_board))
            return num

        # Reset sinking mode if no appropriate moves are found
        self.in_sinking_mode = False
        return self.minmax_ai()  # Switch to MinMax AI if no hits are found
    

    def minmax_ai1(self, depth=2):
        if self.in_sinking_mode:
            return self.basic_ai_MM()

        best_score = -math.inf
        best_move = None
        for move in self.get_available_moves():
            score = self.simulate_and_evaluate(move, depth, False, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is not None:
            num = self.make_move(best_move)

            if self.player1_turn and self.player1.search[best_move] == "H":
                self.in_sinking_mode = True
                self.hit_stack.append(best_move)
            return num
        else:
            return self.basic_ai()  # Fallback to basic_ai if no valid move is found
            
    def basic_ai_MMP(self): 
        
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]

        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)

        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num = self.make_move(u)
                if not self.is_ship_sunk(u):
                    return num
                else:
                    self.in_sinking_mode = False
                    return self.minmax_ai()

        if unknown_with_neighbouring_hits1:
            num = self.make_move(random.choice(unknown_with_neighbouring_hits1))
            if not self.is_ship_sunk(u):
                return num
            else:
                self.in_sinking_mode = False
                return self.minmax_ai()

        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num = self.make_move(random.choice(checker_board))
            if not self.is_ship_sunk(u):
                return num
            else:
                self.in_sinking_mode = False
                return self.minmax_ai()

        # Reset sinking mode if no appropriate moves are found
        self.in_sinking_mode = False
        return self.minmax_ai()  # Switch to MinMax AI if no hits are found
    '''



    '''
    def basic_ai_MM(self):
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]

        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)

        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num = self.make_move(u)
                if not self.is_ship_sunk(num):  # Changed from `u` to `num`
                    return num
                else:
                    self.in_sinking_mode = False
                    return self.minmax_ai()

        if unknown_with_neighbouring_hits1:
            num = self.make_move(random.choice(unknown_with_neighbouring_hits1))
            if not self.is_ship_sunk(num):  # Changed from `u` to `num`
                return num
            else:
                self.in_sinking_mode = False
                return self.minmax_ai()

        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num = self.make_move(random.choice(checker_board))
            if not self.is_ship_sunk(num):  # Changed from `u` to `num`
                return num
            else:
                self.in_sinking_mode = False
                return self.minmax_ai()

        # Reset sinking mode if no appropriate moves are found
        self.in_sinking_mode = False
        return self.minmax_ai()  # Switch to MinMax AI if no hits are found
      
    def minmax_ai(self, depth=2):
        if self.in_sinking_mode:
            return self.basic_ai_MM()

        best_score = -math.inf
        best_move = None
        for move in self.get_available_moves():
            score = self.simulate_and_evaluate(move, depth, False, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is not None:
            num = self.make_move(best_move)

            if self.player1_turn and self.player1.search[best_move] == "H":
                self.in_sinking_mode = True
                self.hit_stack.append(best_move)
            return num
        else:
            return self.basic_ai()  # Fallback to basic_ai if no valid move is found
    

    def is_ship_sunk(self, index):
        player = self.player1 if self.player1_turn else self.player2
        for ship in player.ships:
            if index in ship.indexes:
                for idx in ship.indexes:
                    if player.search[idx] == "U":
                        return False
                return True
        return False
  '''
    ''' 
    def basic_ai_MM(self):
        print("Entering basic_ai_MM")
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]

        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)

        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num = self.make_move(u)
                print(f"Trying move {u}, result: {num}")
                if not self.is_ship_sunk(u):
                    print(f"Ship not sunk at {u}, staying in sinking mode")
                    return num
                else:
                    print(f"Ship sunk at {u}, switching to minmax_ai")
                    self.in_sinking_mode = False
                    return self.minmax_ai()

        if unknown_with_neighbouring_hits1:
            num = self.make_move(random.choice(unknown_with_neighbouring_hits1))
            print(f"Trying move {num} from neighbouring hits 1, result: {num}")
            if not self.is_ship_sunk(num):
                print(f"Ship not sunk at {num}, staying in sinking mode")
                return num
            else:
                print(f"Ship sunk at {num}, switching to minmax_ai")
                self.in_sinking_mode = False
                return self.minmax_ai()

        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num = self.make_move(random.choice(checker_board))
            print(f"Trying move {num} from checkerboard, result: {num}")
            if not self.is_ship_sunk(num):
                print(f"Ship not sunk at {num}, staying in sinking mode")
                return num
            else:
                print(f"Ship sunk at {num}, switching to minmax_ai")
                self.in_sinking_mode = False
                return self.minmax_ai()

        # Reset sinking mode if no appropriate moves are found
        self.in_sinking_mode = False
        print("No appropriate moves found, switching to minmax_ai")
        return self.minmax_ai()  # Switch to MinMax AI if no hits are found
    

    def minmax_ai(self, depth=2):
        print("Entering minmax_ai")
        if self.in_sinking_mode:
            print("Currently in sinking mode")
            return self.basic_ai_MM()

        best_score = -math.inf
        best_move = None
        for move in self.get_available_moves():
            score = self.simulate_and_evaluate(move, depth, False, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is not None:
            num = self.make_move(best_move)
            print(f"Making best move {best_move}, result: {num}")

            if self.player1_turn and self.player1.search[best_move] == "H":
                self.in_sinking_mode = True
                self.hit_stack.append(best_move)
                print(f"Hit at {best_move}, entering sinking mode")
            return num
        else:
            print("No best move found, switching to basic_ai")
            return self.basic_ai()  # Fallback to basic_ai if no valid move is found
    

    def is_ship_sunk(self, index):
        player = self.player1 if self.player1_turn else self.player2
        for ship in player.ships:
            if index in ship.indexes:
                for idx in ship.indexes:
                    if player.search[idx] == "U":
                        return False
                return True
        return False
    
    '''
    '''
    def basic_ai_MM(self):
        print("Entering basic_ai_MM")
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]

        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)

        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num = self.make_move(u)
                print(f"Trying move {u}, result: {num}")
                if not self.is_ship_sunk(u):
                    print(f"Ship not sunk at {u}, staying in sinking mode")
                    return num
                else:
                    print(f"Ship sunk at {u}, resetting sinking mode")
                    self.in_sinking_mode = False

        if unknown_with_neighbouring_hits1:
            num = self.make_move(random.choice(unknown_with_neighbouring_hits1))
            print(f"Trying move {num} from neighbouring hits 1, result: {num}")
            if not self.is_ship_sunk(num):
                print(f"Ship not sunk at {num}, staying in sinking mode")
                return num
            else:
                print(f"Ship sunk at {num}, resetting sinking mode")
                self.in_sinking_mode = False

        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num = self.make_move(random.choice(checker_board))
            print(f"Trying move {num} from checkerboard, result: {num}")
            if not self.is_ship_sunk(num):
                print(f"Ship not sunk at {num}, staying in sinking mode")
                return num
            else:
                print(f"Ship sunk at {num}, resetting sinking mode")
                self.in_sinking_mode = False

        # Reset sinking mode if no appropriate moves are found
        self.in_sinking_mode = False
        print("No appropriate moves found, switching to minmax_ai")
        return self.minmax_ai()  # Switch to MinMax AI if no hits are found
    

    def minmax_ai(self, depth=2):
        print("Entering minmax_ai")
        if self.in_sinking_mode:
            print("Currently in sinking mode")
            return self.basic_ai_MM()

        best_score = -math.inf
        best_move = None
        for move in self.get_available_moves():
            score = self.simulate_and_evaluate(move, depth, False, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is not None:
            num = self.make_move(best_move)
            print(f"Making best move {best_move}, result: {num}")

            if self.player1_turn and self.player1.search[best_move] == "H":
                self.in_sinking_mode = True
                self.hit_stack.append(best_move)
                print(f"Hit at {best_move}, entering sinking mode")
            return num
        else:
            print("No best move found, switching to basic_ai")
            return self.basic_ai()  # Fallback to basic_ai if no valid move is found
    

    def is_ship_sunk(self, index):
        player = self.player1 if self.player1_turn else self.player2
        for ship in player.ships:
            if index in ship.indexes:
                for idx in ship.indexes:
                    if player.search[idx] == "U":
                        return False
                return True
        return False
    '''
    
    
    def basic_ai_MM1(self):
        print("Entering basic_ai_MM")
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]

        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)

        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num = self.make_move(u)
                print(f"Trying move {u}, result: {num}")
                if self.player1.search[u] == "H" and not self.is_ship_sunk(u):
                    print(f"Hit without sink at {u}, staying in sinking mode")
                    self.in_sinking_mode = True
                    return num
                elif self.player1.search[u] == "S":
                    print(f"Ship sunk at {u}, resetting sinking mode")
                    self.in_sinking_mode = False

        if unknown_with_neighbouring_hits1:
            num = self.make_move(random.choice(unknown_with_neighbouring_hits1))
            print(f"Trying move {num} from neighbouring hits 1, result: {num}")
            if self.player1.search[num] == "H" and not self.is_ship_sunk(num):
                print(f"Hit without sink at {num}, staying in sinking mode")
                self.in_sinking_mode = True
                return num
            elif self.player1.search[num] == "S":
                print(f"Ship sunk at {num}, resetting sinking mode")
                self.in_sinking_mode = False

        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num = self.make_move(random.choice(checker_board))
            print(f"Trying move {num} from checkerboard, result: {num}")
            if self.player1.search[num] == "H" and not self.is_ship_sunk(num):
                print(f"Hit without sink at {num}, staying in sinking mode")
                self.in_sinking_mode = True
                return num
            elif self.player1.search[num] == "S":
                print(f"Ship sunk at {num}, resetting sinking mode")
                self.in_sinking_mode = False
        
       

        # Reset sinking mode if no appropriate moves are found
        print("No appropriate moves found, resetting sinking mode and switching to minmax_ai")
        self.in_sinking_mode = False
        return self.minmax_ai()  # Switch to MinMax AI if no hits are found
    
    def basic_ai_MM(self):
    
        print("Entering basic_ai_MM")
        search = self.player1.search if self.player1_turn else self.player2.search
        unknown = [i for i, square in enumerate(search) if square == "U"]
        hits = [i for i, square in enumerate(search) if square == "H"]

        # Check if there are any hits that are not part of sunk ships
        for hit in hits:
            if not self.is_ship_sunk(hit):
                self.in_sinking_mode = True
                break
        else:
            self.in_sinking_mode = False

        unknown_with_neighbouring_hits1 = []
        unknown_with_neighbouring_hits2 = []
        for u in unknown:
            if u + 1 in hits or u - 1 in hits or u + 10 in hits or u - 10 in hits:
                unknown_with_neighbouring_hits1.append(u)
            if u + 2 in hits or u - 2 in hits or u + 20 in hits or u - 20 in hits:
                unknown_with_neighbouring_hits2.append(u)

        for u in unknown:
            if u in unknown_with_neighbouring_hits1 and u in unknown_with_neighbouring_hits2:
                num = self.make_move(u)
                print(f"Trying move {u}, result: {num}")
                if not self.is_ship_sunk(u):
                    print(f"Hit without sink at {u}, staying in sinking mode")
                    self.in_sinking_mode = True
                    return num
                else:
                    print(f"Ship sunk at {u}, resetting sinking mode")
                    self.in_sinking_mode = False
                    return num  # Return the result after sinking the ship

        if unknown_with_neighbouring_hits1:
            num = self.make_move(random.choice(unknown_with_neighbouring_hits1))
            print(f"Trying move {num} from neighbouring hits 1, result: {num}")
            if not self.is_ship_sunk(num):
                print(f"Hit without sink at {num}, staying in sinking mode")
                self.in_sinking_mode = True
                return num
            else:
                print(f"Ship sunk at {num}, resetting sinking mode")
                self.in_sinking_mode = False
                return num  # Return the result after sinking the ship

        checker_board = []
        for u in unknown:
            row = u // 10
            col = u % 10
            if (row + col) % 2 == 0:
                checker_board.append(u)
        if checker_board:
            num = self.make_move(random.choice(checker_board))
            print(f"Trying move {num} from checkerboard, result: {num}")
            if not self.is_ship_sunk(num):
                print(f"Hit without sink at {num}, staying in sinking mode")
                self.in_sinking_mode = True
                return num
            else:
                print(f"Ship sunk at {num}, resetting sinking mode")
                self.in_sinking_mode = False
                return num  # Return the result after sinking the ship
         # Before switching to minmax_ai, check if there are any unsunk hits
        for hit in hits:
             if not self.is_ship_sunk(hit):
                 print(f"Unresolved hit at {hit}, staying in basic_ai_MM")
                 self.in_sinking_mode = True
                 return self.basic_ai_MM()  # Continue in basic_ai_MM if there are unsunk hits

        # Reset sinking mode if no appropriate moves are found
        print("No appropriate moves found, resetting sinking mode and switching to minmax_ai")
        self.in_sinking_mode = False
        return self.minmax_ai()  # Switch to MinMax AI if no hits are found
   
    def minmax_ai(self, depth=2):
        print("Entering minmax_ai")
        if self.in_sinking_mode:
            print("Currently in sinking mode")
            return self.basic_ai_MM()

        best_score = -math.inf
        best_move = None
        for move in self.get_available_moves():
            score = self.simulate_and_evaluate(move, depth, False, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is not None:
            num = self.make_move(best_move)
            print(f"Making best move {best_move}, result: {num}")

            if self.player1_turn and self.player1.search[best_move] == "H":
                self.in_sinking_mode = True
                self.hit_stack.append(best_move)
                print(f"Hit at {best_move}, entering sinking mode")
            return num
        else:
            print("No best move found, switching to basic_ai")
            return self.basic_ai()  # Fallback to basic_ai if no valid move is found
   

    def is_ship_sunk(self, index):
        player = self.player1 if self.player1_turn else self.player2
        for ship in player.ships:
            if index in ship.indexes:
                for idx in ship.indexes:
                    if player.search[idx] == "U":
                        return False
                return True
        return False
       

    def get_available_moves(self):
        search = self.player1.search if self.player1_turn else self.player2.search
        return [i for i, square in enumerate(search) if square == "U" and i not in self.actual_misses]

    def simulate_and_evaluate(self, move, depth, is_maximizing, alpha, beta):
        player = self.player1 if self.player1_turn else self.player2
        opponent = self.player2 if self.player1_turn else self.player1

        # Save current state
        original_search = player.search[:]
        original_player1_turn = self.player1_turn

        # Simulate the move
        assumed_hit = self.assume_hit(player, move, original_search)
        simulated_search = self.simulate_search(player, move, assumed_hit, original_search)

        # Simulate the turn change
        self.player1_turn = not self.player1_turn

        # Evaluate the move
        score = self.minimax(depth - 1, not is_maximizing, alpha, beta, simulated_search)

        # Restore the original state
        self.player1_turn = original_player1_turn

        return score

    def simulate_search(self, player, move, assumed_hit, original_search):
        simulated_search = original_search[:]
        simulated_search[move] = "H" if assumed_hit else "M"
        return simulated_search

    def assume_hit1(self, player, move, search):
        # Enhanced logic for assuming a hit
        row, col = divmod(move, 10)
        score = 0
        
        # Factor 1: Proximity to Known Hits
        for neighbor in self.get_neighbors(move):
            if search[neighbor] == "H":
                score += 10

        # Factor 2: Potential Ship Placements
        potential_placements = self.get_potential_ship_placements(move, search)
        score += potential_placements * 5
        
        # Factor 3: Density of Unknown Cells
        unknown_density = self.get_unknown_density(move, search)
        score += unknown_density * 2

        # Factor 4: Avoid Clustering with Misses
        for neighbor in self.get_neighbors(move):
            if neighbor in self.actual_misses:
                score -= 5

        return score > 10
     
        
     
        
     
    
    def assume_hit(self, player, move, search):
        # Enhanced logic for assuming a hit
        row, col = divmod(move, 10)
        score = 0
        
        
        # Factor 1: Proximity to Known Hits
        for neighbor in self.get_neighbors(move):
            if search[neighbor] == "H":
                if self.is_ship_sunk(neighbor):
                    score -= 5  # Penalty for proximity to a sunk ship
                    #num=self.basic_ai_MM()
                else:
                    score += 90  # Significant positive score for proximity to a hit but not sunk ship
                #score += 10  # Increased weight for proximity to known hits
        # Factor 2: Potential Ship Placements
        potential_placements = self.get_potential_ship_placements(move, search)
        score += potential_placements * 3  # Increased weight for potential ship placements
        
        # Factor 3: Density of Unknown Cells
        unknown_density = self.get_unknown_density(move, search)
        score += unknown_density * 5  # Increased weight for density of unknown cells

        # Factor 4: Avoid Clustering with Misses
        for neighbor in self.get_neighbors(move):
            if neighbor in self.actual_misses:
                score -= 10  # Increased penalty for clustering with misses


        # Factor 6: Strategic Positioning (additional factor)
        if (row + col) % 2 == 0:
            score += 2  # Small bonus for checkerboard pattern positioning

        # Introduce randomness
        random_factor = random.uniform(0, 10)
        score += random_factor

        return score > 15  # Adjust the threshold to accommodate the randomness     

   
    def get_potential_ship_placements(self, move, search):
        count = 0
        row, col = divmod(move, 10)
        directions = [(1, 0), (0, 1)]
        for dr, dc in directions:
            length = 0
            for step in range (5):  # Assuming maximum ship length is 5
                nr, nc = row + dr * step, col + dc * step
                if 0 <= nr < 10 and 0 <= nc < 10 and search[nr * 10 + nc] == "U":
                    length += 1
                else:
                    break
            if length >= 2:  # Consider at least length 2 for potential ship placement
                count += 1
        return count

    def get_unknown_density(self, move, search):
        neighbors = self.get_neighbors(move)
        return sum(1 for neighbor in neighbors if search[neighbor] == "U")

    def evaluate(self, search):
        player_hits = sum(1 for i in search if i == "H")
        opponent_hits = sum(1 for i in self.player2.search if i == "H")
        return player_hits - opponent_hits
    
   


    def minimax(self, depth, is_maximizing, alpha, beta, search):
        if depth == 0 or self.over:
            return self.evaluate(search)
            #return self.evaluate_with_assumed_hit_scores(search)
        
        if is_maximizing:
            max_eval = -math.inf
            for move in self.get_available_moves():
                simulated_search = self.simulate_search(self.player1, move, self.assume_hit(self.player1, move, search), search)
                eval = self.minimax(depth - 1, False, alpha, beta, simulated_search)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in self.get_available_moves():
                simulated_search = self.simulate_search(self.player2, move, self.assume_hit(self.player2, move, search), search)
                eval = self.minimax(depth - 1, True, alpha, beta, simulated_search)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_neighbors(self, move):
        neighbors = []
        row, col = divmod(move, 10)
        if col > 0:
            neighbors.append(move - 1)
        if col < 9:
            neighbors.append(move + 1)
        if row > 0:
            neighbors.append(move - 10)
        if row < 9:
            neighbors.append(move + 10)
        return neighbors
    
    

    


