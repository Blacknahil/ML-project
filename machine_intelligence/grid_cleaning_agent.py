import random
from typing import List
"""
Grid Cleaning Agent Implementation
Complete the agent logic following the specified rules
"""

class GridEnvironment:
    """10x10 grid environment with cleaning simulation"""
    def __init__(self):
        self.size = 10
        self.grid = self._initialize_grid()
        self.agent_position = (0, 0)
        self.performance_score = 0
        self.time_steps = 0
        
    def _initialize_grid(self):
        """Create grid with random dirt and obstacles"""
        ''' clean grid is "0"
            dirt grid is #
            obstacle grid is *
        '''
        grid = [['0' for _ in range(self.size)] for _ in range(self.size)]
        # make some of the grids dirt 
        self._makeRandom(grid,'#', '*', 30)
        self._makeRandom(grid, '*','#', 15) 
        # avoid the first cell to end up on a obstacle cell  
        while grid[0][0] == "*":
            grid[0][0] = '0'
            self._makeRandom(grid, '*','#', 1)
        return grid
    
    def get_agent_perception(self):
        """Return current percept for the agent"""
        x, y = self.agent_position
        percept = {
        'current_cell': self.grid[x][y],
        'adjacent_cells': {
        'north': self._get_cell_state(x-1, y),
        'south': self._get_cell_state(x+1, y),
        'east': self._get_cell_state(x, y+1),
        'west': self._get_cell_state(x, y-1)
        }
        }
        return percept
    
    def _get_cell_state(self, x, y):
        """Helper method to get cell state with boundary checking"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x][y]
        return 'Boundary'
    
    def _makeRandom(self, grid:List[List[str]],replacement:str, other_replacement:str, percent:int):
        replaced_count = 0
        goal = (percent * self.size * self.size) // 100
        while replaced_count < goal:
            x = random.randint(0,self.size -1)
            y = random.randint(0,self.size -1)
            if grid[x][y] != replacement and grid[x][y] != other_replacement:
                grid[x][y] = replacement
                replaced_count += 1
    
    def execute_action(self, action:str):
        """Execute the given action and update environment state"""
        self.time_steps += 1
        x, y = self.agent_position
        if action == 'clean':
            if self.grid[x][y] == '#':
                self.grid[x][y] = '0'
         
        moves = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}
        if action in moves:
            dx, dy = moves[action]
            nx, ny = x + dx, y + dy
            if 0 <=nx < self.size and 0 <= ny < self.size and self.grid[nx][ny] != '*':
                self.agent_position = (nx, ny)
            # else:
                # illegal moves into obstacle or boundary
        

class ReflexCleaningAgent:
    """
    Simple reflex agent for grid cleaning
    Implement the following decision rules:
    1. If current cell is Dirty → Clean
    2. If current cell is Clean → Check adjacent cells
    3. If adjacent dirty cell exists → Move toward it
    4. If no adjacent dirty cells → Move randomly (avoid obstacles)
    5. Never move into obstacles or outside boundaries
    """
    def __init__(self):
        self.action_history = []
    def select_action(self, current_percept):
        """
        Determine next action based on current percept
        Parameters:
        current_percept: Dictionary containing 'current_cell' and 'adjacent_cells'
        Returns:
        String action: 'clean', 'north', 'south', 'east', 'west'
        """
        current_cell = current_percept['current_cell']
        adjacent_cells = current_percept['adjacent_cells']
        
        if current_cell == '#':
            self.action_history.append('clean')
            return 'clean'
        else:
            for direction, state in adjacent_cells.items():
                if state == "#":
                    self.action_history.append(direction)
                    return direction
            # No adjacent dirty cells, move randomly
            # 4 & 5 Move randomly avoiding obstacles and boundaries
            safe_moves = [dir for dir, state in adjacent_cells.items() if state not in ['*', 'Boundary']]
            if safe_moves:
                move =random.choice(safe_moves)
                self.action_history.append(move)
                return move
            else:
                # No safe moves, stay and clean (though should not happen)
                self.action_history.append('stay')
                return 'stay'     

def run_cleaning_simulation(max_steps=200):
    """
    Run the cleaning simulation
    Parameters:
    max_steps: Maximum number of steps to simulate
    Returns:  
    Dictionary containing performance metrics
    """  
    environment = GridEnvironment()
    agent = ReflexCleaningAgent()
    cleaned = 0
    for step in range(max_steps):
        # Simulation loop implementation
        percept = environment.get_agent_perception()
        action = agent.select_action(percept)
        environment.execute_action(action)
        if action == 'clean':
            environment.performance_score += 10
            if percept['current_cell'] == '#':
                cleaned += 1
        elif action in ['north', 'south', 'east', 'west']:
            environment.performance_score -= 1
        else:
            #illegal action
            print ("illegal action taken", action)
            environment.performance_score -= 5
        
        # if step % 20 == 0:
        #     print(f"\nGrid State at Step {step}:")
        #     visualize_grid(environment, step)
    
    return {
    'final_score': environment.performance_score,
    'cells_cleaned': cleaned,
    'steps_taken': environment.time_steps,
    'action history': agent.action_history,
    'environment': environment
    }
    
def visualize_grid(environment, step_number):
    """
    Display the current grid state
    Parameters:
    environment: Current GridEnvironment instance
    step_number: Current simulation step
    Prints formatted grid representation
    """
    symbol_map = {'0': 'C', '#': 'D', '*': 'X'}
    ax,ay = environment.agent_position
    for x,row in enumerate(environment.grid):
        display_row = []
        for y,cell in enumerate(row):
            if (x,y) == (ax,ay):
                display_row.append('A')  # Agent's position
            else:
                display_row.append(symbol_map.get(cell, '?'))
        print(' '.join(display_row))
    
    print(f"\nStep: {step_number} | Performance Score: {environment.performance_score}")
    
def run_multiple_simulations(num_simulations=5, max_steps=200):
    """
    Run multiple cleaning simulations and aggregate results
    Parameters:
    num_simulations: Number of simulations to run
    max_steps: Maximum steps per simulation
    Returns:
    List of dictionaries containing performance metrics for each simulation
    """
    results = []
    seeds = [random.getrandbits(32) for _ in range(num_simulations)]
    for i in range(num_simulations):
        random.seed(seeds[i]) # set different seed for each simulation
        result = run_cleaning_simulation(max_steps = max_steps)
        result["seed"] = seeds[i]
        results.append(result)
    return results 

if __name__ == "__main__":
# Execute simulation and display results
    num_simulations = 5
    results = run_multiple_simulations(num_simulations=num_simulations, max_steps=200)
    performance = 0 
    for result in results:
        performance += result['final_score']
        
        print("Simulation Complete for seed: ", result["seed"]) 
        visualize_grid(result["environment"], result["steps_taken"])
        print(f"Final Performance Score: {result['final_score']}")
        print(f"Cells Cleaned: {result['cells_cleaned']}")
        print(f"Steps Taken: {result['steps_taken']}")
        print()
    
    print(f"Average performace for {num_simulations} simulations is: ", performance/num_simulations)