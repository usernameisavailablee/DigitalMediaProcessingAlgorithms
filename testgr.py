def count_paths(graph, start, end, length):
    # Создание матрицы для хранения количества путей
    paths = [[0 for _ in range(length + 1)] for _ in range(len(graph))]

    # Начальная вершина имеет один путь длины 0 (сама к себе)
    paths[start][0] = 1

    # Подсчёт количества путей длиной i от start к end
    for i in range(1, length + 1):
        for v in range(len(graph)):
            for neighbor in range(len(graph)):
                if graph[v][neighbor]:
                    paths[neighbor][i] += paths[v][i - 1]

    return paths[end][length]

# Пример графа (матрица смежности)
# Замените 1 на 0, если соединение отсутствует, иначе оставьте 1
graph = [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
]

start_vertex = 0  # Индекс вершины X
end_vertex = 3    # Индекс вершины Y1
path_length = 7    # Длина пути

result = count_paths(graph, start_vertex, end_vertex, path_length)
print(f"Количество путей длины {path_length} из вершины X в Y1: {result}")
