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
            res[i + j + 1] += res[i + j] // 10
            res[i + j] = res[i + j] % 10

    res, ind = res[::-1], 0
    while ind < len(res) and res[ind] == 0:
        ind += 1

    res = map(str, res[ind:])
    return "".join(res)


### Letter Combinations of a Phone Number
# Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.
# A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

from collections import deque
from typing import List

def letterCombinations(self, digits: str) -> List[str]:
        ans = []
        if len(digits) < 1:
            return ans

        q = deque()
        q.append("")
        chars = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]

        for d in digits:
            digit = int(d)
            qSize = len(q)
            while qSize > 0:
                front = q.popleft()
                for char in chars[digit]:
                    q.append(front + char)
                qSize -= 1

        return list(q)


### Substring with Concatenation of All Words
# You are given a string s and an array of strings words. All the strings of words are of the same length.
# A concatenated string is a string that exactly contains all the strings of any permutation of words concatenated.


def findSubstring(s, words):
    ans = []
    if not s or not words:
        return ans

    freqMap = {}
    for word in words:
        freqMap[word] = freqMap.get(word, 0) + 1

    n = len(s)
    wordCount = len(words)
    wordSize = len(words[0])
    windowSize = wordCount * wordSize

    for offset in range(wordSize):
        left = offset
        curr = {}
        count = 0

        for right in range(offset, n, wordSize):
            word = s[right : right + wordSize]

            if word in freqMap:
                curr[word] = curr.get(word, 0) + 1
                count += 1

                while curr[word] > freqMap[word]:
                    leftWord = s[left : left + wordSize]
                    curr[leftWord] -= 1
                    left += wordSize
                    count -= 1

                if count == wordCount:
                    ans.append(left)
                    leftWord = s[left : left + wordSize]
                    curr[leftWord] -= 1
                    left += wordSize
                    count -= 1
            else:
                curr.clear()
                count = 0
                left = right + wordSize
    return ans
