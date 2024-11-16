
def insertion_srot(array : list)->list:
    for i in range(1 ,len(array)):
        j=i
        while(j>0 and array[j-1] > array[j]):
            temp = array[j]
            array[j]=array[j-1]
            array[j-1] = temp
            j=j-1  
    return array        

def merge_sort(array : list)-> list:
    if(len(array)==1):
        return array
    
    half_way = len(array)//2
    right_array = array[:half_way]
    left_array = array[half_way:]
    right_array = merge_sort(right_array)
    left_array = merge_sort(left_array)
    
    return merge(left_array , right_array)

def merge(left : list , right : list)->list:
    array = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            array.append(left[left_index])
            left_index+=1
        else:
            array.append(right[right_index])  
            right_index+=1
    array.extend(left[left_index:])  
    array.extend(right[right_index:])      
    return array          
            
            
        
print(merge_sort([1,2,6,5,3,0]))
print(insertion_srot([1,2,6,5,3,0]))