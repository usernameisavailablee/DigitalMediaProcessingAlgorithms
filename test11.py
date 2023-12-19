from itertools import permutations

# Создаем список всех четырехзначных чисел
numbers = [int(''.join(p)) for p in permutations('123456789', 4)]

# Создаем список всех семи чисел, сумма которых равна 10578
sum_numbers = list(permutations(numbers, 7))
sum_numbers = [num for num in sum_numbers if sum(num) == 10578]

# Вычисляем сумму остальных чисел
other_numbers = sum(set(numbers) - set(sum_numbers[0]))

print(other_numbers)
