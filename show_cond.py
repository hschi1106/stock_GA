import json

if __name__ == "__main__":
    # CONSTANTS
    chromosome_set = 'new_espx'
    cond_list = json.load(open(f"chromosomes/{chromosome_set}_cond.json", "r"))
    
    # print("Condition list:", cond_list)
    
    # Load the best chromosome
    best_chromosome = json.load(open(f"chromosomes/{chromosome_set}.json", "r"))
    
    # Display the best chromosome
    # print("Best Chromosome:", best_chromosome)
    
    # Show the conditions
    for cidx, ch in enumerate(best_chromosome[:5]):
        print("Chromosome: ", cidx)
        for idx, condition in enumerate(cond_list):
            if ch[idx] == 1:
                # Display the condition if it's part of the chromosome
                print(f"\tCondition {idx}: {condition[0]} > {condition[1]} * {condition[2]}")
            elif ch[idx] == -1:
                # Display the condition if it's negated in the chromosome
                print(f"\tCondition {idx}: {condition[0]} < {condition[1]} * {condition[2]}")