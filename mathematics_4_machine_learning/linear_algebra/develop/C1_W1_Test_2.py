import numpy as np

def test_system(b=6, m=1, c=3):
    print(f'For b={b}, m={m}, c={c}:')
    print(f'{2 * b + 1 * m + 5 * c} = 20, {2 * b + 1 * m + 5 * c == 20}')
    print(f'{1 * b + 2 * m + 1 * c} = 10, {1 * b + 2 * m + 1 * c == 10}')
    print(f'{2 * b + 1 * m + 3 * c} = 15, {2 * b + 1 * m + 3 * c == 15}')


test_system(6, 1, 3)
test_system(2.5, 2.5, 5.5)
test_system(2.5, 2.5, 2.5)
test_system(1.5, 3.5, 2.5)


array1 = np.array([1, 2, 3, 0, 2, 2, 1, 4, 5], dtype=int).reshape(3, 3)

print(np.linalg.det(array1))


