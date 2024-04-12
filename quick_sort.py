def quick_sort(arr, start, end):
    if start < end:
        mid = partition(arr, start, end)
        quick_sort(arr, start, mid-1)
        quick_sort(arr, mid+1, end)
    return arr

def partition(arr, start, end):
    pivot = arr[start]
    left = start + 1
    right = end
    while True:
        while left<=right and arr[left] <= pivot:
            left += 1
        while left<=right and arr[right] > pivot:
            right -= 1
        if left>=right:
            break
        arr[left], arr[right] = arr[right], arr[left]
    arr[start], arr[right] = arr[right], arr[start]
    return right

if __name__ == "__main__":
    test_data = [
        [1,1,2,2,3,1],
        [1,43,2,4,6,7,2,4],
        [6,4,2,13,1]
    ]
    for a in test_data:
        sorted_a = quick_sort(a, 0, len(a)-1)
        print(sorted_a)
