def string_multiply(num1, num2):
    if num1 =='0' or num2=='0':
        return '0'

    m,n = len(num1),len(num2)
    ans = [0]*(m+n)
    for i in range(m-1, -1, -1):
        a = int(num1[i])
        for j in range(n-1, -1, -1):
            b = int(num2[j])
            c = a*b + ans[i+j+1]
            ans[i+j+1] = c%10
            ans[i+j] += c//10
    i = 0 
    while ans[i] == 0:
        i += 1
    return ''.join(str(t) for t in ans[i:])


if __name__ == '__main__':
    a = '1234'
    b = '123'
    gold_answer = int(a)*int(b)
    print('gold_answer: ', gold_answer)
    print('str_answer: ', string_multiply(a,b))
