def binary_search(array : list , target ):
    left = 0
    right = len(array)-1
    while(left <= right):
        mid = (right+left)//2
        if array[mid] == target:
            return mid
        elif array[mid] < target:
            left = mid +1
        else :
            right = mid -1    
print(binary_search([0,2,3,4,5,6,7,8,9,10] ,6))            