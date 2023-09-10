

with open("./logs/0.15/distil/student.txt") as file:
    test_acc = 0
    idx = 0
    for line in file.readlines():
        if(idx>609):
            if(idx%3 == 0):
                # print(line[38:44])
                test_acc = max(test_acc, float(line[38:44]))
        idx += 1
                
print(test_acc)