#!/bin/bash

# # 0.15 exp
# python3 main.py -s 0.15 -se 0 -c 0 > ./logs/0.15/teacher_0.txt & python3 main.py -s 0.15 -se 1 -c 0 > ./logs/0.15/teacher_1.txt & python3 main.py -s 0.15 -se 2 -c 0 > ./logs/0.15/teacher_2.txt & python3 main.py -s 0.15 -se 0 -c 1 -t 0 > ./logs/0.15/student_0.txt & python3 main.py -s 0.15 -se 1 -c 1 -t 0 > ./logs/0.15/student_1.txt & python3 main.py -s 0.15 -se 2 -c 1 -t 0 > ./logs/0.15/student_2.txt

# if [ $? -eq 0 ]; then
#     # 0.20 exp
#     python3 main.py -s 0.20 -se 0 -c 2 > ./logs/0.20/teacher_0.txt & python3 main.py -s 0.20 -se 1 -c 2 > ./logs/0.20/teacher_1.txt & python3 main.py -s 0.20 -se 2 -c 2 > ./logs/0.20/teacher_2.txt & python3 main.py -s 0.20 -se 0 -c 3 -t 0 > ./logs/0.20/student_0.txt & python3 main.py -s 0.20 -se 1 -c 3 -t 0 > ./logs/0.20/student_1.txt & python3 main.py -s 0.20 -se 2 -c 3 -t 0 > ./logs/0.20/student_2.txt

#     if [ $? -eq 0 ]; then
#         # 0.25 exp
#         python3 main.py -s 0.25 -se 0 -c 0 > ./logs/0.25/teacher_0.txt & python3 main.py -s 0.25 -se 1 -c 0 > ./logs/0.25/teacher_1.txt & python3 main.py -s 0.25 -se 2 -c 0 > ./logs/0.25/teacher_2.txt & python3 main.py -s 0.25 -se 0 -c 1 -t 0 > ./logs/0.25/student_0.txt & python3 main.py -s 0.25 -se 1 -c 1 -t 0 > ./logs/0.25/student_1.txt & python3 main.py -s 0.25 -se 2 -c 1 -t 0 > ./logs/0.25/student_2.txt

#         if [ $? -eq 0 ]; then
#             # 0.30 exp
#             python3 main.py -s 0.30 -se 0 -c 2 > ./logs/0.30/teacher_0.txt & python3 main.py -s 0.30 -se 1 -c 2 > ./logs/0.30/teacher_1.txt & python3 main.py -s 0.30 -se 2 -c 2 > ./logs/0.30/teacher_2.txt & python3 main.py -s 0.30 -se 0 -c 3 -t 0 > ./logs/0.30/student_0.txt & python3 main.py -s 0.30 -se 1 -c 3 -t 0 > ./logs/0.30/student_1.txt & python3 main.py -s 0.30 -se 2 -c 3 -t 0 > ./logs/0.30/student_2.txt

#             if [ $? -eq 0 ]; then
#                 # 0.35 exp
#                 python3 main.py -s 0.35 -se 0 -c 0 > ./logs/0.35/teacher_0.txt & python3 main.py -s 0.35 -se 1 -c 0 > ./logs/0.35/teacher_1.txt & python3 main.py -s 0.35 -se 2 -c 0 > ./logs/0.35/teacher_2.txt & python3 main.py -s 0.35 -se 0 -c 1 -t 0 > ./logs/0.35/student_0.txt & python3 main.py -s 0.35 -se 1 -c 1 -t 0 > ./logs/0.35/student_1.txt & python3 main.py -s 0.35 -se 2 -c 1 -t 0 > ./logs/0.35/student_2.txt

#             else
#                 echo "failed"
#             fi

#         else
#             echo "failed"
#         fi

#     else
#         echo "failed"
#     fi

# else
#     echo "failed"
# fi

# distil
python3 distil.py -s 0.15 -c 0 -se 0 > ./logs/0.15/distil/student_0.txt & python3 distil.py -s 0.20 -c 0 -se 0 > ./logs/0.20/distil/student_0.txt & python3 distil.py -s 0.25 -c 1 -se 0 > ./logs/0.25/distil/student_0.txt & python3 distil.py -s 0.30 -c 2 -se 0 > ./logs/0.30/distil/student_0.txt & python3 distil.py -s 0.35 -c 3 -se 0 > ./logs/0.35/distil/student_0.txt
if [ $? -eq 0 ]; then
    python3 distil.py -s 0.15 -c 0 -se 1 > ./logs/0.15/distil/student_1.txt & python3 distil.py -s 0.20 -c 0 -se 1 > ./logs/0.20/distil/student_1.txt & python3 distil.py -s 0.25 -c 1 -se 1 > ./logs/0.25/distil/student_1.txt & python3 distil.py -s 0.30 -c 2 -se 1 > ./logs/0.30/distil/student_1.txt & python3 distil.py -s 0.35 -c 3 -se 1 > ./logs/0.35/distil/student_1.txt
    if [ $? -eq 0 ]; then
        python3 distil.py -s 0.15 -c 0 -se 2 > ./logs/0.15/distil/student_2.txt & python3 distil.py -s 0.20 -c 0 -se 2 > ./logs/0.20/distil/student_2.txt & python3 distil.py -s 0.25 -c 1 -se 2 > ./logs/0.25/distil/student_2.txt & python3 distil.py -s 0.30 -c 2 -se 2 > ./logs/0.30/distil/student_2.txt & python3 distil.py -s 0.35 -c 3 -se 2 > ./logs/0.35/distil/student_2.txt
    else
        echo "Failed"
    fi
else
    echo "Failed"
fi