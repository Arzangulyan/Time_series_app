# расстояние между точками перегиба
# сшивка (учет выпуклости/вогнутости + экстремумов)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')


interval = 24
priority = 0
delta = 6


df = pd.read_excel('/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/Attempt_1/Bizin_morfology/SO2_vse_urovni_Vyetnam_1000.xlsx')
print("data import complete")
y = np.array(df.S1CO2)[0:1000]
x = np.array(df.index)[0:1000]
size = len(x)
x = x-x[0]


conv_up_dp = np.zeros((size+1, size+1), dtype=int)

conv_down_dp = np.zeros((size+1, size+1), dtype=int)

evolution = np.zeros(size, dtype=int)-1  # 0-decreases; 1-increases; -1-difficult to say
increases = []
decreases = []
for i in range(1, size-2):
    if f.mono_incr(y[i-1:i+1+1]):
        evolution[i] = 1
    elif f.mono_decr(y[i-1:i+1+1]):
        evolution[i] = 0


pseudo_evolution = evolution.copy()
for i in range(1, size-1):
    if pseudo_evolution[i] == -1:
        j = f.find_next(pseudo_evolution, i)
        if j and pseudo_evolution[j] == pseudo_evolution[i-1]:
            while pseudo_evolution[i] == -1:
                pseudo_evolution[i] = pseudo_evolution[j]
                i += 1
        else:
            i = j
            continue

increases = np.where(pseudo_evolution == 1)[0]
decreases = np.where(pseudo_evolution == 0)[0]

if increases[0] < decreases[0]:  # if function increases in the beginning
    switch_parameter = False
else:
    switch_parameter = True
    y = -y

    indices_one = evolution == 1
    indices_zero = evolution == 0
    evolution[indices_one] = 0
    evolution[indices_zero] = 1

    indices_one = pseudo_evolution == 1
    indices_zero = pseudo_evolution == 0
    pseudo_evolution[indices_one] = 0
    pseudo_evolution[indices_zero] = 1

    increases, decreases = decreases, increases


first_max = np.argmax(y[0:2*interval])

chosen_infl_p = [first_max - interval - interval//2]
good_infl_p = []
j = 0
APPROXIMATION = np.array([])

while True:
    j += 1

    if (j % 2):

        prev_infl_p = chosen_infl_p[-1]
        search_time_interval = x[max(prev_infl_p + interval-delta, 1):prev_infl_p+interval+delta+1]
        search_segment = np.intersect1d(search_time_interval, increases)
        segment_limit = min(prev_infl_p+2*interval-delta, size)
        if j == 1:
            chosen_infl_p = [0]
            prev_infl_p = 0

        if not search_segment.any():
            break
        delts = [[], []]

        for i in search_segment:
            convexdownapp = f.projconvdown(y, prev_infl_p, i, conv_down_dp)
            convexupapp = f.projconvup(y, i, segment_limit, conv_up_dp)
            app = np.append(convexdownapp, convexupapp)
            if (len(convexdownapp) >= 2 and len(convexupapp) >= 2
                and f.checkconvexdown(np.concatenate((convexdownapp[-2:], convexupapp[:1])))
                    and not f.is_ext(np.concatenate((convexdownapp[-1:], convexupapp[:2])), 1)):
                delts[0].append((i, f.dif(y[prev_infl_p:segment_limit], app)))
            else:
                delts[1].append((i, f.dif(y[prev_infl_p:segment_limit], app)))

        chosen_infl_p.append(f.optimal_point(delts, priority))
        APPROXIMATION = np.append(APPROXIMATION, f.projconvdown(
            y, chosen_infl_p[-2], chosen_infl_p[-1], conv_down_dp))

        if (f.checkconvexup(APPROXIMATION[chosen_infl_p[-2]-2:chosen_infl_p[-2]+1])
                and not f.is_ext(APPROXIMATION, chosen_infl_p[-2])):
            good_infl_p.append(chosen_infl_p[-2])

    else:

        prev_infl_p = chosen_infl_p[-1]
        search_segment = x[prev_infl_p+interval-delta:prev_infl_p+interval+delta+1]
        segment_limit = min(prev_infl_p+2*interval-delta, size)

        if not search_segment.any():
            break
        delts = [[], []]

        for i in search_segment:
            convexupapp = f.projconvup(y, prev_infl_p, i, conv_up_dp)
            convexdownapp = f.projconvdown(y, i, segment_limit, conv_down_dp)
            app = np.append(convexupapp, convexdownapp)
            if (len(convexdownapp) >= 2 and len(convexupapp) >= 2
                and f.checkconvexup(np.concatenate((convexupapp[-2:], convexdownapp[:1])))
                    and not f.is_ext(np.concatenate((convexupapp[-1:], convexdownapp[:2])), 1)):
                delts[0].append((i, f.dif(y[prev_infl_p:segment_limit], app)))
            else:
                delts[1].append((i, f.dif(y[prev_infl_p:segment_limit], app)))

        chosen_infl_p.append(f.optimal_point(delts, priority))
        delts = [[], []]
        APPROXIMATION = np.append(
            APPROXIMATION, f.projconvup(y, chosen_infl_p[-2], chosen_infl_p[-1], conv_up_dp))

        if (f.checkconvexdown(APPROXIMATION[chosen_infl_p[-2]-2:chosen_infl_p[-2]+1])
                and not f.is_ext(APPROXIMATION, chosen_infl_p[-2])):
            good_infl_p.append(chosen_infl_p[-2])


if not j % 2:
    APPROXIMATION = np.append(APPROXIMATION, f.projconvup(y, chosen_infl_p[-1], size, conv_up_dp))
    if (f.checkconvexdown(APPROXIMATION[chosen_infl_p[-1]-2:chosen_infl_p[-1]+1])
            and not f.is_ext(APPROXIMATION[chosen_infl_p[-1]-1:chosen_infl_p[-1]+2], 1)):
        good_infl_p.append(chosen_infl_p[-1])
else:
    APPROXIMATION = np.append(APPROXIMATION, f.projconvdown(y, chosen_infl_p[-1], size, conv_down_dp))
    if (f.checkconvexup(APPROXIMATION[chosen_infl_p[-1]-2:chosen_infl_p[-1]+1])
            and not f.is_ext(APPROXIMATION[chosen_infl_p[-1]-1:chosen_infl_p[-1]+2], 1)):
        good_infl_p.append(chosen_infl_p[-1])

if switch_parameter:
    APPROXIMATION = -APPROXIMATION
    y = -y
    increases, decreases = decreases, increases

    indices_one = evolution == 1
    indices_zero = evolution == 0
    evolution[indices_one] = 0
    evolution[indices_zero] = 1

    indices_one = pseudo_evolution == 1
    indices_zero = pseudo_evolution == 0
    pseudo_evolution[indices_one] = 0
    pseudo_evolution[indices_zero] = 1

chosen_infl_p.remove(0)


fig, ax1 = plt.subplots(1)
ax1.plot(x, y, 'ro', markersize=2, label='experimental data')
ax1.plot(x, APPROXIMATION, 'b', linewidth=1, label='result of morphological filtering')
ax1.plot(x[chosen_infl_p], APPROXIMATION[chosen_infl_p], 'go', markersize=5)
ax1.plot(x[good_infl_p], APPROXIMATION[good_infl_p], 'yo', markersize=5)
ax1.set_xlabel("Time (1 tick = 30 min)")
ax1.set_ylabel("CO2 concentration")
legend1 = ax1.legend(loc='best', shadow=True, fontsize='large')
# ax2.plot(
#     x, y-APPROXIMATION, 'g', linewidth=1, label="residual series"
#     )
# ax2.set_xlabel("Time (1 tick = 30 min)")
# ax2.set_ylabel("CO2 concentration")
# legend2 = ax2.legend(loc='best', shadow=True, fontsize='large')
