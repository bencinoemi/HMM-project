class DatasetCreator:
    def __init__(self, file):
        self.file = file

    def separating_string(self, string):
        return string.replace('\n', '').split(',')

    def count_sensors(self, line):
        return len(self.separating_string(self.file[line + 1]))

    def create_new_lines(self, line, new_data, n_newlines):
        return new_data.extend([self.file[line] for i in range(n_newlines)])

    def create_file(self, data):
        created = ''
        for temp in data:
            for item in temp:
                created = created+item
        return created

    def create_dataset(self):
        count = 0
        data = []
        temp = []
        for line in range(len(self.file)):
            if count == 0:
                temp = []
                nrow = self.count_sensors(line)
                self.create_new_lines(line, temp, nrow)
            else:
                objects = self.separating_string(self.file[line])
                for obj in range(nrow):
                    temp[obj] = temp[obj][:len(temp[obj])-1] + ',' + objects[obj] + '\n'
                if count == 4:
                    data.append(temp)
                    count = -1
            count += 1
        return self.create_file(data)


import os
os.chdir('/home/noe/Università/in_corso/Machine Learning/progetto/dataset')

file_sub1 = open(r'subject1/activities_data.csv', "r+").readlines()
open(r'subject1/activities_data.csv', "w+").write(DatasetCreator(file_sub1).create_dataset())

file_sub2 = open(r'subject2/activities_data.csv', "r+").readlines()
open('subject2/activities_data.csv', 'w+').write(DatasetCreator(file_sub2).create_dataset())



