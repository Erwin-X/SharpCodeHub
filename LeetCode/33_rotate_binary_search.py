def rotate_binary_search(nums, target):
    l,r = 0, len(nums)
    while l<r:
        mid = l + (r-l)//2
        if nums[mid] == target:
            return mid
        
        if nums[l]<=nums[mid]:
            if nums[l]<=target<nums[mid]:
                r = mid
            else:
                l = mid+1
        else:
            if nums[mid]<=target<nums[r-1]:
                l = mid+1
            else:
                r = mid
    return -1


if __name__ == '__main__':
    # nums = [4,5,6,7,0,1,2];target = 0
    nums = [4,5,6,7,0,1,2];target = 3
    print(rotate_binary_search(nums, target))
