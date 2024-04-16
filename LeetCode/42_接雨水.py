def trap(height):
    """
    :type height: List[int]
    :rtype: int
    """
    sum_water = 0
    l, r = 1, len(height)-2
    maxl, maxr = height[0], height[-1]
    while l<=r:
        if maxl<=maxr:
            cur_water = max(maxl-height[l], 0)
            maxl = max(maxl, height[l])
            l += 1
        else:
            cur_water = max(maxr-height[r], 0)
            maxr = max(maxr, height[r])
            r -= 1
        sum_water += cur_water
    return sum_water

if __name__ == "__main__":
    test_h = [[0,1,0,2,1,0,1,3,2,1,2,1],    # result: 6
              [4,2,0,3,2,5]]                # result: 9
    
    for h in test_h:
        print(trap(h))