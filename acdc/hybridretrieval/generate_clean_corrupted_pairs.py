import random

# Define possible names, countries, cities, and capitals
names = ["Alice", "Bob", "John", "Peter", "Lucy", "Michael", "Sara", "Tom", "Anna", "David"]
countries = ["France", "Germany", "USA", "Turkey", "Italy", "Spain", "Canada", "Australia", "Japan", "Brazil"]
cities = ["Paris", "Berlin", "Washington", "Ankara", "Rome", "Madrid", "Toronto", "Canberra", "Tokyo", "Rio de Janeiro"]
capitals = ["Paris", "Berlin", "Washington", "Ankara", "Rome", "Madrid", "Toronto", "Canberra", "Tokyo", "Brasilia"]

# Function to generate clean prompts
def generate_clean_prompts():
    clean_prompts = []
    for _ in range(20):
        # Randomly select unique persons
        P1, P2, P3 = random.sample(names, 3)
        # Randomly select unique countries
        C1, C2, C3 = random.sample(countries, 3)
        # Randomly select capitals corresponding to the countries
        K1, K2, K3 = random.sample(capitals, 3)

        prompt = f"{P1} lives in {C1}, {K1} - {P1}, {P2} lives in {C2}, {K2} - {P2}, {P3} lives in {C3}, {K3} - {P3}"
        clean_prompts.append((prompt, f"{K3} - {P3}"))

    return clean_prompts

# Generate clean prompts
clean_prompts = generate_clean_prompts()

print(clean_prompts)

# Function to generate corrupted prompts
def generate_corrupted_prompts(clean_prompts):
    corrupted_prompts = []
    for prompt, query in clean_prompts:
        parts = prompt.split(", ")
        p3_info = parts[2]
        print(parts)
        print(p3_info)
        P1, C1, K1 = parts[0].split(" ")[0], parts[0].split(" ")[2], parts[0].split(" ")[3]
        P2, C2, K2 = parts[1].split(" ")[0], parts[1].split(" ")[2], parts[1].split(" ")[3]
        P3, C3, K3 = p3_info.split(" ")[0], p3_info.split(" ")[2], p3_info.split(" ")[3]

        # Corrupt capital 3
        corrupted_K3 = random.choice([K1, K2] + random.sample(capitals, 1))

        # Corrupt person 3
        corrupted_P3 = random.choice([P1, P2] + random.sample(names, 1))

        # Corrupt country 3
        corrupted_C3 = random.choice(random.sample(countries, 1))

        # Create corrupted prompt
        corrupted_prompt1 = f"{P1} lives in {C1}, {K1} - {P1}, {P2} lives in {C2}, {K2} - {P2}, {P3} lives in {corrupted_C3}, {K3} - {P3}"
        corrupted_prompt2 = f"{P1} lives in {C1}, {K1} - {P1}, {P2} lives in {C2}, {K2} - {P2}, {P3} lives in {C3}, {corrupted_K3} - {P3}"
        corrupted_prompt3 = f"{P1} lives in {C1}, {K1} - {P1}, {P2} lives in {C2}, {K2} - {P2}, {P3} lives in {C3}, {K3} - {corrupted_P3}"
        corrupted_prompts.append(corrupted_prompt1, corrupted_prompt2, corrupted_prompt3)

    return corrupted_prompts

# Generate corrupted prompts
corrupted_prompts = generate_corrupted_prompts(clean_prompts)

# Display clean and corrupted prompts
for idx, (clean_prompt, corrupted_prompt) in enumerate(zip(clean_prompts, corrupted_prompts)):
    print(f"{idx+1}. Clean Prompt: {clean_prompt}")
    print(f"   Corrupted Prompt: {corrupted_prompt}")