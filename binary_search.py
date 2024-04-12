def binary_search(arr, target):
    left,right = 0,len(arr)-1
    while left <= right:
        mid = (left+right)//2
        if arr[mid] == target:
            return mid
        elif arr[mid]<target:
            left = mid+1
        else:
            right = mid-1
    return -1

if __name__ == "__main__":
    test_data = [
        [1,2,3,4,5],
        [2,3,4,6,12,43,46,100],
        [43,577,1234,5355,32556,212455]
    ]
    targets = [3, 43, 100, 32556]
    for arr in test_data:
        for t in targets:
            idx = binary_search(arr, t)
            print(idx)
        print("\n")
