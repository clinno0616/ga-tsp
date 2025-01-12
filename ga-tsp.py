import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

class TSPGA:
    def __init__(self, num_cities=30, pop_size=200, elite_size=20, mutation_rate=0.01):
        self.num_cities = num_cities
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        
        # Fixed cities coordinates
        self.cities = np.array([
            [200, 1500], [1000, 2000], [800, 2000], [400, 1500], [600, 2100],  # 0-4
            [600, 2000], [1400, 1500], [1600, 1800], [1000, 1900], [800, 1600],  # 5-9
            [600, 1700], [400, 1700], [1800, 2200], [400, 1500], [200, 1300],  # 10-14
            [600, 1500], [1200, 1500], [200, 1600], [800, 1500], [1000, 1500],  # 15-19
            [600, 2000], [400, 2000], [400, 1600], [1600, 2200], [1200, 1400],  # 20-24
            [400, 2100], [1400, 1600], [1200, 1600], [1000, 2000], [300, 2100]  # 25-29
        ])
        
        # Initialize population with valid routes
        self.population = []
        for _ in range(pop_size):
            # Create a random permutation of all cities
            route = np.arange(num_cities)
            np.random.shuffle(route)
            self.population.append(route)
        
        self.best_route = None
        self.best_distance = float('inf')
        self.generation = 0
        self.history = []

    def validate_route(self, route):
        """Validate if a route contains all cities exactly once"""
        if route is None or len(route) != self.num_cities:
            return False
        # 檢查是否包含所有城市且每個城市只出現一次
        city_set = set(route)
        return len(city_set) == self.num_cities and len(route) == self.num_cities

    def calculate_distance(self, route):
        """Calculate total distance of a route"""
        total_distance = 0
        for i in range(len(route)):
            from_city = self.cities[route[i]]
            to_city = self.cities[route[(i + 1) % self.num_cities]]
            total_distance += np.sqrt(np.sum((from_city - to_city) ** 2))
        return total_distance

    def fitness(self, route):
        """Calculate fitness of a route (inverse of distance)"""
        return 1 / self.calculate_distance(route)

    def rank_routes(self):
        """Rank all routes in population by fitness"""
        fitness_results = {}
        for i in range(len(self.population)):
            fitness_results[i] = self.fitness(self.population[i])
        return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)

    def selection(self, ranked_population):
        """Select routes for breeding using roulette wheel selection"""
        selection_results = []
        
        # 保留精英個體
        for i in range(self.elite_size):
            selection_results.append(ranked_population[i][0])
        
        # 計算累積概率
        fitness_sum = sum([item[1] for item in ranked_population])
        probabilities = [item[1]/fitness_sum for item in ranked_population]
        cum_probabilities = np.cumsum(probabilities)
        
        # 輪盤賭選擇
        for _ in range(len(ranked_population) - self.elite_size):
            pick = random.random()
            for i, cum_prob in enumerate(cum_probabilities):
                if pick <= cum_prob:
                    selection_results.append(ranked_population[i][0])
                    break
        
        return selection_results

    def mating_pool(self, selection_results):
        """Create mating pool from selected routes"""
        pool = []
        for i in range(len(selection_results)):
            index = selection_results[i]
            pool.append(self.population[index])
        return pool

    def breed(self, parent1, parent2):
        """使用次序交叉(Order Crossover, OX)產生子代"""
        size = self.num_cities
        child = np.full(size, -1)  # 初始化子代為-1
        
        # 隨機選擇交叉區段
        start, end = sorted(random.sample(range(size), 2))
        
        # 直接從parent1複製選定區段
        child[start:end+1] = parent1[start:end+1]
        
        # 從parent2填充剩餘位置
        parent2_idx = 0
        for i in range(size):
            if child[i] == -1:  # 需要填充的位置
                while parent2[parent2_idx] in child:  # 找到一個未使用的城市
                    parent2_idx += 1
                child[i] = parent2[parent2_idx]
        
        # 驗證子代
        if not self.validate_route(child):
            print("Warning: Invalid child generated, using random route instead")
            child = np.arange(size)
            np.random.shuffle(child)
            
        return child

    def breed_population(self, mating_pool):
        """Breed entire population"""
        children = []
        
        # Keep elite routes
        for i in range(self.elite_size):
            children.append(mating_pool[i])
        
        # Breed the rest
        pool = random.sample(mating_pool, len(mating_pool))
        for i in range(len(mating_pool) - self.elite_size):
            child = self.breed(pool[i], pool[len(mating_pool) - 1 - i])
            children.append(child)
            
        return children

    def mutate(self, individual):
        """使用交換突變"""
        mutated = individual.copy()
        
        # 根據突變率決定突變次數
        num_mutations = int(self.num_cities * self.mutation_rate) + 1
        
        # 執行多次交換突變
        for _ in range(num_mutations):
            # 隨機選擇兩個不同位置
            idx1, idx2 = random.sample(range(self.num_cities), 2)
            # 交換
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        # 驗證突變後的路徑
        if not self.validate_route(mutated):
            print("Warning: Invalid mutation, using original route")
            return individual
            
        return mutated

    def mutate_population(self, population):
        """Mutate entire population"""
        mutated_pop = []
        
        # Keep elite routes without mutation
        for i in range(self.elite_size):
            mutated_pop.append(population[i])
            
        # Mutate the rest
        for ind in population[self.elite_size:]:
            mutated_ind = self.mutate(ind)
            mutated_pop.append(mutated_ind)
            
        return mutated_pop

    def next_generation(self):
        """產生下一代族群"""
        # 評估並排序當前族群
        ranked_pop = self.rank_routes()
        
        # 選擇父代
        selection_results = self.selection(ranked_pop)
        mating_pool = self.mating_pool(selection_results)
        
        # 產生子代
        children = self.breed_population(mating_pool)
        
        # 突變
        self.population = self.mutate_population(children)
        
        # 更新最佳解
        current_best_idx = ranked_pop[0][0]
        current_best_route = self.population[current_best_idx]
        current_best_distance = self.calculate_distance(current_best_route)
        
        # 驗證最佳解
        if not self.validate_route(current_best_route):
            print("Warning: Invalid best route detected")
            return
            
        if current_best_distance < self.best_distance:
            self.best_distance = current_best_distance
            self.best_route = current_best_route.copy()
        
        self.generation += 1
        self.history.append(self.best_distance)

    def display_final_result(self):
        """Display the final result in a new window"""
        plt.figure(figsize=(10, 8))
        
        # Plot the cities
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100)
        
        # Plot the best route
        route_coords = self.cities[self.best_route]
        route_coords = np.vstack((route_coords, route_coords[0]))  # Connect back to start
        plt.plot(route_coords[:, 0], route_coords[:, 1], 'b-', linewidth=2, alpha=0.8)
        
        # Add city numbers
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(f'City {i}', (x, y), xytext=(5, 5), textcoords='offset points')
            
        plt.title(f'Final Best Route (Distance: {self.best_distance:.2f})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Show the plot in a new window
        plt.show()

    def run(self, num_generations=500, convergence_limit=50):
        """Run the genetic algorithm"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # For convergence check
        no_improvement_count = 0
        last_best_distance = float('inf')
        
        def update(frame):
            nonlocal no_improvement_count, last_best_distance
            
            self.next_generation()
            
            # Check for convergence
            if abs(self.best_distance - last_best_distance) < 0.1:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            last_best_distance = self.best_distance
            
            # Clear the axes
            ax1.clear()
            ax2.clear()
            
            # Plot current best route
            route_coords = self.cities[self.best_route]
            route_coords = np.vstack((route_coords, route_coords[0]))  # Connect back to start
            
            ax1.plot(route_coords[:, 0], route_coords[:, 1], 'b-')
            ax1.scatter(self.cities[:, 0], self.cities[:, 1], c='red')
            ax1.set_title(f'Best Route (Generation {self.generation})\nDistance: {self.best_distance:.2f}')
            
            # Plot progress
            ax2.plot(self.history)
            ax2.set_title('Best Distance Over Time')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Distance')
            
            plt.tight_layout()
            
            # Stop if converged
            if no_improvement_count >= convergence_limit:
                print(f"\nConverged after {self.generation} generations!")
                print(f"Best route found with distance: {self.best_distance:.2f}")
                print("Route:", self.best_route.tolist())
                print(f"Number of unique cities visited: {len(set(self.best_route))}")
                
                # Close the animation window
                plt.close(fig)
                
                # Display the final result in a new window
                self.display_final_result()
                
                anim.event_source.stop()
        
        anim = FuncAnimation(fig, update, frames=num_generations, interval=50, repeat=False)
        plt.show()

# 使用示例
if __name__ == "__main__":
    ga = TSPGA(num_cities=30, pop_size=200, elite_size=20, mutation_rate=0.01)
    ga.run(num_generations=500, convergence_limit=50)