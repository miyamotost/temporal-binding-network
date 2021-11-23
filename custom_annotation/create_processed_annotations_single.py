import sys

args = sys.argv
if len(args) == 2:
    input_name = args[1]
else:
    print('invalid args.')
    exit()

f_con = open('./annotations/label.txt', 'r', encoding='UTF-8')
labels = {}
for line in f_con:
    line = line.replace('\n', '')
    line = line.split(', ')
    labels[line[1]] = line[0]
f_con.close()

f_details = open('./annotations/details/{}'.format(input_name), 'r', encoding='UTF-8')
f_pro = open('./annotations/processed/{}'.format(input_name), 'w', encoding='UTF-8')
for line in f_details:
    line = line.replace('\n', '')
    line = line.split(', ')

    l_fps = line[13].replace('fps:', '')
    l_fps = int(l_fps)
    l_start = line[5].split('.')[0].split(':')
    l_start = int(l_start[0])*60*60*l_fps + int(l_start[1])*60*l_fps + int(l_start[2])*l_fps
    line[7] = str(l_start)
    l_end = line[6].split('.')[0].split(':')
    l_end = int(l_end[0])*60*60*l_fps + int(l_end[1])*60*l_fps + int(l_end[2])*l_fps
    line[8] = str(l_end)
    line[9] = line[3] #verb
    line[10] = labels[line[3]] #label

    f_pro.write('{}\n'.format(', '.join(line)))
f_details.close()
f_pro.close()
