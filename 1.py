def count_vowels(s: str) -> int:
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count


def all_unique(s: str) -> bool:
    return len(s) == len(set(s))

print(count_vowels("Hello World"))  
print(all_unique("Hello"))         