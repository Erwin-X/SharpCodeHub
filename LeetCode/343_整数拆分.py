def integer_break(n):
    dp = [0] * (n+1)
    dp[1] = 1

    for i in range(2, n+1):
        for j in range(1, i//2 + 1):
            dp[i] = max(dp[i], j*(i-j), j*dp[i-j])

    return dp[-1]

if __name__ == "__main__":
    test_x = [10, 20]
    for x in test_x:
        print(x, integer_break(x))
