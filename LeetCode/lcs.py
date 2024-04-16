def lcs(X, Y):
    m,n = len(X),len(Y)
    L = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    length = L[m][n]
    subseq_list = [""]*length
    i, j = m,n
    while length>0:
        if X[i-1] == Y[j-1]:
            subseq_list[length-1] = X[i-1]
            i -= 1
            j -= 1
            length -= 1
        elif L[i][j-1] > L[i-1][j]:
            j -= 1
        else:
            i -= 1
    subseq = "".join(subseq_list)
    return subseq


if __name__  == "__main__":
    X = "ABCDEFGs2XX"
    Y = "ACEG2X2AX"
    print(X)
    print(Y)
    print(lcs(X,Y))
