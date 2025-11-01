#### Given a string s, find the length of the longest substring without duplicate characters.
s = "bbbbb"

def longestSubString(s):
    sSet = set()
    right = 0
    left = 0
    n = len(s)
    maxLen = 0

    while right < n:
        if s[right] not in sSet:
            sSet.add(s[right])
            maxLen = max(maxLen, right - left + 1)
            right += 1
        else:
            sSet.remove(s[left])
            left += 1

    return maxLen

# print(longestSubString(s))

# Median of Two Sorted Arrays
# Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
# The overall run time complexity should be O(log (m+n)).

def medianOftwoArrays(arr1, arr2): # time complexity : O(n.logn)
    arr1.extend(arr2)
    arr1.sort()
    n = len(arr1)
    mid = n // 2

    if n % 2 == 0:
        return (arr1[mid - 1] + arr1[mid]) / 2
    else:
        return arr1[mid]


def findMedianSortedArrays(nums1, nums2) -> float: # time complexity : O(log (m+n))
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    total = len(nums1) + len(nums2)
    half = total // 2

    l , r = 0, len(nums1) - 1

    while True:
        i = (l + r) // 2
        j = half - i - 2

        Aleft = nums1[i] if i >= 0 else float("-inf")
        Aright = nums1[i+1] if (i + 1) < len(nums1) else float("inf")
        Bleft = nums2[j] if j >= 0 else float("-inf")
        Bright = nums2[j+1] if (j + 1) < len(nums2) else float("inf")

        if Aleft <= Bright and Bleft <= Aright:
            if total % 2:
                return min(Aright, Bright)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2

        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1

    
nums1 = [1,2]
nums2 = [3,4]

print("Median: ", medianOftwoArrays(nums1, nums2))


### 3Sum Closest
# Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.
# Return the sum of the three integers.
# You may assume that each input would have exactly one solution.

def threeSumClosest(nums, target) -> int:
    nums.sort()
    n = len(nums)
    closestSum = float('inf')

    for i in range(n-2):
        l, r = i + 1, len(nums) - 1
            
        while l < r:
            threeSum = nums[i] + nums[l] + nums[r]

            if abs(threeSum - target) < abs(closestSum - target):
                closestSum = threeSum

            if threeSum > target:
                r -= 1
            elif threeSum < target:
                l += 1
            else:
                return threeSum
                
    return closestSum