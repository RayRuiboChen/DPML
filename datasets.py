import csv
import datetime
import os
import torch
import random


class one_task():
    def __init__(self, path, data_name, batch_size, code):
        '''
        data_name: 'five_minute' or 'hourly'
        '''
        self.batch_size = batch_size
        self.data_name = data_name
        self.path = path
        self.code = code
        if data_name != 'five_minute' and data_name != 'ten_minute':
            raise NotImplementedError

        train_dataset, dev_dataset, test_dataset = self.read_csv()
        if self.data_name == 'five_minute':
            self.trainloader = self.get_loader_five_minute(raw_data=train_dataset,
                                                           batch_size=self.batch_size, mode='train')
            self.devloader = self.get_loader_five_minute(raw_data=dev_dataset,
                                                         batch_size=self.batch_size, mode='dev')
            self.testloader = self.get_loader_five_minute(raw_data=test_dataset,
                                                          batch_size=self.batch_size, mode='test')
        elif self.data_name == 'ten_minute':
            self.trainloader = self.get_loader_ten_minute(raw_data=train_dataset,
                                                          batch_size=self.batch_size, mode='train')
            self.devloader = self.get_loader_ten_minute(raw_data=dev_dataset,
                                                        batch_size=self.batch_size, mode='dev')
            self.testloader = self.get_loader_ten_minute(raw_data=test_dataset,
                                                         batch_size=self.batch_size, mode='test')

    def read_csv(self):
        train_data, train_history, test_data, test_history = {}, set(), {}, set()
        finetune_data, finetune_history, dev_data, dev_history = {}, set(), {}, set()

        with open(self.path) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == '':
                    continue

                time_stamp = datetime.datetime.strptime(row[2] + ' ' + row[6], "%Y-%m-%d %H:%M")

                chlov = [float(row[1]), float(row[3]), float(row[4]), float(row[5]), float(row[7])]

                if self.data_name == 'five_minute':
                    if time_stamp.year > 2018:
                        continue
                    if time_stamp.year < 2017:
                        continue
                    if time_stamp.year == 2017 and time_stamp.month < 11:
                        continue
                    if time_stamp.year == 2018 and time_stamp.month > 2:
                        continue

                    # 2017 Dec 15 18 19 20 12 22 25 26
                    # dev 27 28 29
                    # 2018 Feb 1 2 5 test

                    if time_stamp.year == 2018:
                        test_data[time_stamp] = chlov
                        test_history.add(datetime.datetime.strptime(row[2], "%Y-%m-%d"))
                    else:
                        dev_data[time_stamp] = chlov
                        dev_history.add(datetime.datetime.strptime(row[2], "%Y-%m-%d"))

                        train_data[time_stamp] = chlov
                        train_history.add(datetime.datetime.strptime(row[2], "%Y-%m-%d"))

                elif self.data_name == 'ten_minute':

                    if time_stamp.year > 2018:
                        continue
                    if time_stamp.year < 2017:
                        continue
                    if time_stamp.year == 2017 and time_stamp.month < 6:
                        continue
                    if time_stamp.year == 2018 and time_stamp.month > 2:
                        continue

                    # 2017 8 9 10 11train
                    # 2017 12 dev
                    # 2018 2 test

                    if time_stamp.year == 2018:
                        test_data[time_stamp] = chlov
                        test_history.add(datetime.datetime.strptime(row[2], "%Y-%m-%d"))
                    else:
                        if time_stamp.month > 9:
                            dev_data[time_stamp] = chlov
                            dev_history.add(datetime.datetime.strptime(row[2], "%Y-%m-%d"))

                        if time_stamp.month < 12:
                            train_data[time_stamp] = chlov
                            train_history.add(datetime.datetime.strptime(row[2], "%Y-%m-%d"))

        return (train_data, train_history), (dev_data, dev_history), (test_data, test_history)

    def get_loader_five_minute(self, raw_data, batch_size, mode):
        '''
        d: train_data or test_data
        history_set: train_history or test_history
        '''
        d, history_set = raw_data
        dataloader = []
        for x in d:
            if mode == 'train':
                if (x.year < 2017):
                    continue
                if (x.year == 2017) and (x.month != 12):
                    continue
                if (x.year == 2017) and (x.month == 12) and (x.day < 15):
                    continue
                if (x.year == 2017) and (x.month == 12) and (x.day > 26):
                    continue
                train_mode = True

            elif mode == 'dev':
                if (x.year < 2017):
                    continue
                if (x.year == 2017) and (x.month != 12):
                    continue
                if (x.year == 2017) and (x.month == 12) and (x.day < 27):
                    continue
                if (x.year == 2017) and (x.month == 12) and (x.day > 29):
                    continue
                train_mode = False

            elif mode == 'test':
                if (x.year == 2018) and (x.month != 2):
                    continue
                if (x.year == 2018) and (x.month == 2) and (x.day > 5):  # 共3个交易日
                    continue
                train_mode = False

            else:
                raise NotImplementedError

            # 2017 Dec 15 18 19 20 21 22 25 26 train
            # 2017 Dec 27 28 29 dev
            # 2018 Feb 1 2 5 test

            tensor = torch.FloatTensor(1, 12, 5)
            tensor[:, :, :] = -1

            test_time = x + datetime.timedelta(minutes=-5 * 12)
            if test_time not in d:
                continue

            con_flag = False
            for i in range(1, 13):
                y = x + datetime.timedelta(minutes=-5 * i)
                if y in d:
                    for j in range(5):
                        tensor[0, 12 - i, j] = d[y][j]
                else:
                    con_flag = True
                    break
            if con_flag:
                continue

            history = torch.FloatTensor(1, 20, 5)
            history[:, :, :] = -1
            cnt = 0
            y = x
            while (cnt < 20):
                y = y + datetime.timedelta(days=-1)
                if (datetime.datetime.strptime(str(y).split(' ')[0], "%Y-%m-%d") in history_set) \
                        or (y.year < x.year):
                    cnt += 1
                    if y in d:
                        for j in range(5):
                            history[0, 20 - cnt, j] = d[y][j]
                    else:
                        con_flag = True
                        break
            if con_flag:
                continue

            if (history.min().item() > -0.5) and (tensor.min().item() > -0.5):
                dataloader.append((tensor, history, torch.FloatTensor([[d[x][-1]]])))

        random.shuffle(dataloader)
        l = len(dataloader)
        testloader = []

        if l == 0:
            return testloader

        if train_mode == True:
            tensor = torch.cat([dataloader[j][0] for j in range(l)], dim=0)
            tensor_v20 = torch.cat([dataloader[j][1] for j in range(l)], dim=0)
            tensor_v = torch.cat([dataloader[j][2] for j in range(l)], dim=0)
            input_data = torch.cat([tensor.reshape(l, -1), tensor_v20.reshape(l, -1)], dim=1)
            input_data = torch.log(1 + input_data)
            tensor_v = torch.log(1 + tensor_v)
            testloader = (input_data, tensor_v)
        else:
            for i in range(l // batch_size):
                tensor = torch.cat([dataloader[j][0] for j in range(i * batch_size, (i + 1) * batch_size)], 0)
                tensor_v20 = torch.cat([dataloader[j][1] for j in range(i * batch_size, (i + 1) * batch_size)], 0)
                tensor_v = torch.cat([dataloader[j][2] for j in range(i * batch_size, (i + 1) * batch_size)], 0)

                input_data = torch.cat([tensor.reshape(batch_size, -1), tensor_v20.reshape(batch_size, -1)], dim=1)
                input_data = torch.log(1 + input_data)
                tensor_v = torch.log(1 + tensor_v)
                testloader.append((input_data, tensor_v))
            if l % batch_size != 0:
                tensor = torch.cat([dataloader[j][0] for j in range(-(l % batch_size), 0)], 0)
                tensor_v20 = torch.cat([dataloader[j][1] for j in range(-(l % batch_size), 0)], 0)
                tensor_v = torch.cat([dataloader[j][2] for j in range(-(l % batch_size), 0)], 0)

                input_data = torch.cat([tensor.reshape(l % batch_size, -1), tensor_v20.reshape(l % batch_size, -1)],
                                       dim=1)
                input_data = torch.log(1 + input_data)
                tensor_v = torch.log(1 + tensor_v)
                testloader.append((input_data, tensor_v))

        return testloader

    def get_loader_ten_minute(self, raw_data, batch_size, mode):
        '''
        d: train_data or test_data
        history_set: train_history or test_history
        '''
        d, history_set = raw_data
        dataloader = []
        for x in d:

            if mode == 'train':
                if (x.year < 2017):
                    continue
                if (x.year == 2017) and (x.month == 12):
                    continue
                if (x.year == 2017) and (x.month < 8):
                    continue
                train_mode = True

            elif mode == 'dev':
                if (x.year < 2017):
                    continue
                if (x.year == 2017) and (x.month != 12):
                    continue
                train_mode = False

            elif mode == 'test':
                if (x.year != 2018):
                    continue
                if (x.year == 2018) and (x.month != 2):
                    continue
                train_mode = False

            else:
                raise NotImplementedError

            test_time = x + datetime.timedelta(minutes=-5 * 25)
            if test_time not in d:
                continue

            if x.hour > 11:
                if x.hour < 14:
                    continue
                if x.hour == 14:
                    if x.minute < 35:
                        continue

            tensor = torch.FloatTensor(1, 25, 5)
            tensor[:, :, :] = -1

            con_flag = False
            for i in range(1, 26):
                y = x + datetime.timedelta(minutes=-5 * i)
                if y in d:
                    for j in range(5):
                        tensor[0, 25 - i, j] = d[y][j]
                else:
                    con_flag = True
                    break

            if con_flag:
                continue

            history1 = torch.FloatTensor(1, 20, 5)
            history1[:, :, :] = -1
            cnt = 0
            y = x
            while (cnt < 20):
                y = y + datetime.timedelta(days=-1)
                if (datetime.datetime.strptime(str(y).split(' ')[0], "%Y-%m-%d") in history_set) \
                        or (y.year < x.year):
                    cnt += 1
                    if y in d:
                        for j in range(5):
                            history1[0, 20 - cnt, j] = d[y][j]
                    else:
                        con_flag = True
                        break
            if con_flag:
                continue

            history2 = torch.FloatTensor(1, 20, 5)
            history2[:, :, :] = -1
            cnt = 0
            y = x + datetime.timedelta(minutes=-5)
            while (cnt < 20):
                y = y + datetime.timedelta(days=-1)
                if (datetime.datetime.strptime(str(y).split(' ')[0], "%Y-%m-%d") in history_set) or (y.year < x.year):
                    cnt += 1
                    if y in d:
                        for j in range(5):
                            history2[0, 20 - cnt, j] = d[y][j]
                    else:
                        con_flag = True
                        break
            if con_flag:
                continue

            if (history1.min().item() > -0.5) and (tensor.min().item() > -0.5) and (history2.min().item() > -0.5):
                dataloader.append(
                    self.process_ten_minute_data(history1, history2, tensor, torch.FloatTensor([[d[x][-1]]]))
                )

        random.shuffle(dataloader)
        l = len(dataloader)
        testloader = []

        if l == 0:
            return testloader

        if train_mode == True:
            tensor = torch.cat([dataloader[j][0] for j in range(l)], dim=0)
            tensor_v20 = torch.cat([dataloader[j][1] for j in range(l)], dim=0)
            tensor_v = torch.cat([dataloader[j][2] for j in range(l)], dim=0)
            input_data = torch.cat([tensor.reshape(l, -1), tensor_v20.reshape(l, -1)], dim=1)
            input_data = torch.log(1 + input_data)
            tensor_v = torch.log(1 + tensor_v)
            testloader = (input_data, tensor_v)

        else:
            for i in range(l // batch_size):
                tensor = torch.cat([dataloader[j][0] for j in range(i * batch_size, (i + 1) * batch_size)], 0)
                tensor_v20 = torch.cat([dataloader[j][1] for j in range(i * batch_size, (i + 1) * batch_size)], 0)
                tensor_v = torch.cat([dataloader[j][2] for j in range(i * batch_size, (i + 1) * batch_size)], 0)

                input_data = torch.cat([tensor.reshape(batch_size, -1), tensor_v20.reshape(batch_size, -1)], dim=1)
                input_data = torch.log(1 + input_data)
                tensor_v = torch.log(1 + tensor_v)
                testloader.append((input_data, tensor_v))
            if l % batch_size != 0:
                tensor = torch.cat([dataloader[j][0] for j in range(-(l % batch_size), 0)], 0)
                tensor_v20 = torch.cat([dataloader[j][1] for j in range(-(l % batch_size), 0)], 0)
                tensor_v = torch.cat([dataloader[j][2] for j in range(-(l % batch_size), 0)], 0)

                input_data = torch.cat([tensor.reshape(l % batch_size, -1), tensor_v20.reshape(l % batch_size, -1)],
                                       dim=1)
                input_data = torch.log(1 + input_data)
                tensor_v = torch.log(1 + tensor_v)
                testloader.append((input_data, tensor_v))

        return testloader

    def process_ten_minute_data(self, history1, history2, tensor, v):
        '''
        history1: [1,20,5]
        history2: [1,20,5]
        tensor: [1,25,5]
        v:[1]
        '''
        res_history = torch.FloatTensor(1, 20, 5)
        res_history[:, :, 0] = history1[:, :, 0]  # close
        res_history[:, :, 1] = torch.max(history1[:, :, 1], history2[:, :, 1])  # high
        res_history[:, :, 2] = torch.min(history1[:, :, 2], history2[:, :, 2])  # low
        res_history[:, :, 3] = history2[:, :, 3]  # open
        res_history[:, :, 4] = history1[:, :, 4] + history2[:, :, 4]  # volume

        res_v = v + tensor[0, -1, -1]

        res_tensor = torch.FloatTensor(1, 12, 5)
        for i in range(12):
            res_tensor[:, i, 0] = tensor[:, 2 * i + 1, 0]  # close
            res_tensor[:, i, 1] = torch.max(tensor[:, i * 2, 1], tensor[:, i * 2 + 1, 1])  # high
            res_tensor[:, i, 2] = torch.min(tensor[:, i * 2, 2], tensor[:, i * 2 + 1, 2])  # low
            res_tensor[:, i, 3] = tensor[:, 2 * i, 3]  # open
            res_tensor[:, i, 4] = tensor[:, 2 * i, 4] + tensor[:, 2 * i + 1, 4]  # volume

        return res_tensor, res_history, res_v


class Tasks():
    def __init__(self, data_name, filelist, dir_name='./topix500', cache_dir='./smaller_cache', batch_size=32):
        self.data_name = data_name  # 'five_minute' or 'hourly'
        self.filelist = filelist
        self.cache_dir = cache_dir
        self.dir_name = dir_name
        self.tasks = {}
        self.task_names = set()
        self.batch_size = batch_size

        self.get_all_tasks()

    def get_all_tasks(self):
        print("getting all tasks...")
        show_step = 10
        for idx, task_name in enumerate(self.filelist):
            path = task_name + '.csv'
            self.task_names.add(task_name)
            cache_name = os.path.join(self.cache_dir, task_name + '_' + self.data_name + '.pt')
            if os.path.exists(cache_name):
                self.tasks[task_name] = torch.load(cache_name)
            else:
                self.tasks[task_name] = one_task(path=os.path.join(self.dir_name, path),
                                                 data_name=self.data_name, batch_size=self.batch_size, code=task_name)
                torch.save(self.tasks[task_name], cache_name)
                if (idx + 1) % show_step == 0:
                    print(f'step {idx + 1} finished!')
        print('finish getting tasks!')

    def checkdata(self):
        file_cnt = len(self.filelist)
        train_cnt, dev_cnt, test_cnt = 0, 0, 0
        train_zero, dev_zero, test_zero = 0, 0, 0
        train_total, dev_total, test_total = 0, 0, 0

        for i in self.filelist:
            if len(self.tasks[i].trainloader) > 0:
                train_len = self.tasks[i].trainloader[0].shape[0]
                train_total += train_len
            else:
                train_len = 0
            dev_len = 0
            test_len = 0

            if len(self.tasks[i].devloader) > 0:
                for x, y in self.tasks[i].devloader:
                    dev_len += x.shape[0]

            if len(self.tasks[i].testloader) > 0:
                for x, y in self.tasks[i].testloader:
                    test_len += x.shape[0]

            dev_total += dev_len
            test_total += test_len

            if train_len < 32:
                train_cnt += 1
                if train_len == 0:
                    train_zero += 1
            if test_len < 8:
                test_cnt += 1
                if len(self.tasks[i].testloader) == 0:
                    test_zero += 1

            if dev_len < 8:
                dev_cnt += 1
                if len(self.tasks[i].devloader) == 0:
                    dev_zero += 1
        print('total stock number:', file_cnt)
        print(train_total, train_cnt, train_zero)
        print(dev_total, dev_cnt, dev_zero)
        print(test_total, test_cnt, test_zero)
