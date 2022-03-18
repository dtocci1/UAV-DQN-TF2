'''
Create a balanced CSV of all data for quantization process
100-1000 images
'''
import random
import csv

with open('./dataset/sorted/benign.csv') as file:
    benign = list(csv.reader(file))
with open('./dataset/sorted/bot.csv') as file:
    bot = list(csv.reader(file))
with open('dataset/sorted/bruteforce.csv') as file:
    bruteforce = list(csv.reader(file))
with open('dataset/sorted/ddos.csv') as file:
    ddos = list(csv.reader(file))
with open('dataset/sorted/ftp.csv') as file:
    ftp = list(csv.reader(file))
with open('dataset/sorted/portscan.csv') as file:
    portscan = list(csv.reader(file))
with open('dataset/sorted/sql.csv') as file:
    sql = list(csv.reader(file))
with open('dataset/sorted/ssh.csv') as file:
    ssh = list(csv.reader(file))
with open('dataset/sorted/xss.csv') as file:
    xss = list(csv.reader(file))
with open('dataset/sorted/dos.csv') as file:
    dos = list(csv.reader(file))

def gen_frame(attack):
    action_rewards = {0: "benign", # Match actions with rewards
        1: "portscan",
        2: "ddos",
        3: "bot",
        4: "bruteforce",
        5: "xss",
        6: "sql",
        7: "ftp",
        8: "ssh",
        9: "dos"}

    if action_rewards[attack] == "benign":
        index = random.randint(0,len(benign)-1)
        frame = benign[index]
    elif action_rewards[attack] == "bot":
        index = random.randint(0,len(bot)-1)
        frame = bot[index]
    elif action_rewards[attack] == "bruteforce":
        index = random.randint(0,len(bruteforce)-1)
        frame = bruteforce[index]
    elif action_rewards[attack] == "ddos":
        index = random.randint(0,len(ddos)-1)
        frame = ddos[index]
    elif action_rewards[attack] == "ftp":
        index = random.randint(0,len(ftp)-1)
        frame = ftp[index]
    elif action_rewards[attack] == "portscan":
        index = random.randint(0,len(portscan)-1)
        frame = portscan[index]
    elif action_rewards[attack] == "sql":
        index = random.randint(0,len(sql)-1)
        frame = sql[index]
    elif action_rewards[attack] == "ssh":
        index = random.randint(0,len(ssh)-1)
        frame = ssh[index]
    elif action_rewards[attack] == "xss":
        index = random.randint(0,len(xss)-1)
        frame = xss[index]
    elif action_rewards[attack] == "dos":
        index = random.randint(0,len(dos)-1)
        frame = dos[index]

    return frame

def main():
    size = 100 # 100-1000 images to generate
    data = []

    for i in range(size):
        atype = random.randint(0,9)
        frame = gen_frame(atype)
        data.append(frame)

    with open('dataset/quantize/test.csv','w') as file:
        write = csv.writer(file)
        write.writerows(data)


if __name__ == '__main__':
    main()