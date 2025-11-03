### Multiply Strings
# Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
# Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.

# def multiply(num1: str, num2: str) -> str:
#     a=int(num1)
#     b=int(num2)
#     c=a*b 
#     d=str(c)
#     return d

def multiply(num1: str, num2: str) -> str:
    if "0" in [num1, num2]:
        return "0"

    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    num1, num2 = num1[::-1], num2[::-1]

    for i in range(m):
        for j in range(n):
            digit = int(num1[i]) * int(num2[j])
            res[i + j] += digit
            res[i + j + 1] += (res[i + j] // 10)
            res[i + j] = (res[i + j] % 10)

    res, ind = res[::-1], 0
    while ind < len(res) and res[ind] == 0:
        ind += 1

    res = map(str, res[ind:])
    return "".join(res)