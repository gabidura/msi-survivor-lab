from msi_survivor import windowed_survivor_count

H = (0,)
z = 8
n = 58  # Goldbach sum N = n + 2 = 60
result = windowed_survivor_count(H, z, n, 0, 59)
print(result)

