def edit_distance(str1, str2):
    m,n = len(str1),len(str2)
    distances = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        distances[i][0] = i
    for j in range(n+1):
        distances[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                cost = 0
            else:
                cost = 1
            distances[i][j] = min(distances[i-1][j] + 1,     # 删除
                                  distances[i][j-1] + 1,     # 插入
                                  distances[i-1][j-1] + cost)   # 替换
    return distances[m][n]

if __name__  == "__main__":
    X = "abcd"
    Y = "abdcd"
    print(X)
    print(Y)
    print(edit_distance(X,Y))
