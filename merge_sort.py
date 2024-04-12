def merge_sort(arr):
    if len(arr)<2:
        return arr
    mid = len(arr)//2
    L = arr[:mid]
    R = arr[mid:]
    merge_sort(L)
    merge_sort(R)

    # merge_two_sorted_arrs
    i=j=k=0
    while i<len(L) and j<len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1
    return arr

if __name__ == "__main__":
    test_data = [
        [1,1,2,2,3,1],
        [1,43,2,4,6,7,2,4],
        [6,4,2,13,1]
    ]
    for a in test_data:
        sorted_a = merge_sort(a)
        print(sorted_a)