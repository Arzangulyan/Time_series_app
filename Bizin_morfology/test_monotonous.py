import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')


interval = 24
delta = 6
priority = 0


df = pd.read_excel('/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/Attempt_1/Bizin_morfology/SO2_vse_urovni_Vyetnam_1000.xlsx')
print("data import complete")
y = np.array(df.S1CO2)[0:1000]
x = np.array(df.index)[0:1000]
size = len(x)
x = x-x[0]

incr_dp = np.zeros((size+1, size+1), dtype=int)
decr_dp = np.zeros((size+1, size+1), dtype=int)

first_min = np.argmin(y[0:2*interval])
first_max = np.argmax(y[0:2*interval])

switch_parameter = False
if first_min < first_max:
    y = -y
    switch_parameter = True
    first_min, first_max = first_max, first_min

minima = []  # 0-decreases; 1-increases; 2-difficult to say
maxima = []
for i in range(max(1, first_max-delta), size-1):
    if f.is_max(y, i):
        maxima.append(i)
    elif f.is_min(y, i):
        minima.append(i)


chosen_extrema = [first_max-interval]
good_extrema = []
j = 0
APPROXIMATION = np.array([])


while True:
    j += 1
    if (j % 2):

        prev_extremum = chosen_extrema[-1]
        search_segment = x[max(prev_extremum+interval-delta, 1):prev_extremum+interval+delta+1]
        segment_limit = min(prev_extremum+2*interval-delta, size)
        if j == 1:
            chosen_extrema = [0]
            prev_extremum = 0

        if not search_segment.any():
            break
        delts = [[], []]

        for i in search_segment:
            incr_app = f.proj_mono_incr(y, prev_extremum, i, incr_dp)
            decr_app = f.proj_mono_decr(y, i, segment_limit, decr_dp)
            app = np.append(incr_app, decr_app)
            if (len(incr_app) >= 2 and len(decr_app) >= 2 and f.is_max(np.concatenate((incr_app[-1:], decr_app[:2])), 1)
                    and f.mono_incr(np.concatenate((incr_app[-2:], decr_app[:1])))):
                delts[0].append((i, f.dif(y[prev_extremum:segment_limit], app)))
            else:
                delts[1].append((i, f.dif(y[prev_extremum:segment_limit], app)))

        chosen_extrema.append(f.optimal_point(delts, priority))
        APPROXIMATION = np.append(APPROXIMATION, f.proj_mono_incr(y, chosen_extrema[-2], chosen_extrema[-1], incr_dp))

        if f.is_min(APPROXIMATION, chosen_extrema[-2]):
            good_extrema.append(chosen_extrema[-2])

    else:

        prev_extremum = chosen_extrema[-1]
        search_segment = x[prev_extremum+interval-delta:prev_extremum+interval+delta+1]
        segment_limit = min(prev_extremum+2*interval-delta, size)

        if not search_segment.any():
            break
        delts = [[], []]

        for i in search_segment:
            decr_app = f.proj_mono_decr(y, prev_extremum, i, decr_dp)
            incr_app = f.proj_mono_incr(y, i, segment_limit, incr_dp)
            app = np.append(decr_app, incr_app)
            if (len(incr_app) >= 2 and len(decr_app) >= 2 and f.is_min(np.concatenate((decr_app[-1:], incr_app[:2])), 1)
                    and f.mono_decr(np.concatenate((decr_app[-2:], incr_app[:1])))):
                delts[0].append((i, f.dif(y[prev_extremum:segment_limit], app)))
            else:
                delts[1].append((i, f.dif(y[prev_extremum:segment_limit], app)))

        chosen_extrema.append(f.optimal_point(delts, priority))
        APPROXIMATION = np.append(APPROXIMATION, f.proj_mono_decr(
            y, chosen_extrema[-2], chosen_extrema[-1], decr_dp))

        if f.is_max(APPROXIMATION, chosen_extrema[-2]):
            good_extrema.append(chosen_extrema[-2])


if j % 2:
    APPROXIMATION = np.append(APPROXIMATION, f.proj_mono_incr(y, chosen_extrema[-1], size, incr_dp))
    if f.is_min(APPROXIMATION, chosen_extrema[-1]):
        good_extrema.append(chosen_extrema[-1])
else:
    APPROXIMATION = np.append(APPROXIMATION, f.proj_mono_decr(y, chosen_extrema[-1], size, decr_dp))
    if f.is_max(APPROXIMATION, chosen_extrema[-1]):
        good_extrema.append(chosen_extrema[-1])

if switch_parameter:
    APPROXIMATION = -(APPROXIMATION)
    y = -y
    first_min, first_max = first_max, first_min

chosen_extrema.remove(0)


fig, ax1 = plt.subplots(1)
ax1.plot(x, y, 'ro', markersize=4, label='experimental data')
ax1.plot(x, APPROXIMATION, 'b', linewidth=1, label='result of morphological filtering')
ax1.plot(x[chosen_extrema], APPROXIMATION[chosen_extrema], 'go', markersize=10, label='chosen extrema')
ax1.plot(x[good_extrema], APPROXIMATION[good_extrema], 'yo', markersize=10, label='good extrema')
ax1.set_xlabel("Time, 1 tick = 30 min")
ax1.set_ylabel("CO2 concentration, ppm")
legend1 = ax1.legend(loc='best', shadow=True, fontsize='large')

# fig, ax2 = plt.subplots(1)
# ax2.plot(x, y-APPROXIMATION, 'g', linewidth=1, label="residual series")
# ax2.set_xlabel("Time, 1 tick = 30 min")
# ax2.set_ylabel("CO2 concentration, ppm")
# legend2 = ax2.legend(loc='best', shadow=True, fontsize='large')

# PERIODS = []
# for i in range(0, len(chosen_extrema)-2):
#     PERIODS.append((chosen_extrema[i+2]-chosen_extrema[i])/2)

# fig, ax = plt.subplots(1)
# ax.plot(np.linspace(0, len(PERIODS)-1, len(PERIODS)), PERIODS, 'g', linewidth=1, label='peroids')
# ax.plot(np.linspace(0, len(PERIODS)/2, int(len(PERIODS)/2)+1),
#         [np.mean(PERIODS)]*(int(len(PERIODS)/2)+1), 'b', linewidth=1, label='periods average')
# ax.set_ylabel('period, hours')
# legend1 = ax.legend(loc='best', shadow=True, fontsize='large')
