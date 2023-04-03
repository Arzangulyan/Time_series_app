import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_morfology as f
import heapq


# parameters settings
period = 48
delta = 3


# data import
df = pd.read_excel('/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/Attempt_1/Bizin_morfology/SO2_vse_urovni_Vyetnam_1000.xlsx')
print("data import complete")
y = np.array(df.S1CO2)[11:200]
x = np.array(df.index)[11:200]
x = x-x[0]

size = len(x)
form_parameters = np.linspace(0, size, size*2//period + 1, dtype=int)
inner_cycle = 2 * delta + 1
outer_cycle = len(form_parameters) - 2
for i in range(1, len(form_parameters)-1):
    h = []
    outer_done = i - 1

    for infl_p in range(form_parameters[i-1] + period//2 - delta, form_parameters[i-1] + period//2 + delta + 1):
        infl_p = min(infl_p, size)
        inner_done = infl_p - (form_parameters[i-1] + period//2 - delta)
        print(round((outer_done * inner_cycle + inner_done) / (outer_cycle * inner_cycle) * 100, 2), '%')

        form_parameters[i] = infl_p

        A = f.convex_constraints_matrix(form_parameters)
        app = f.numeric_sol(y, A)
        heapq.heappush(h, (np.linalg.norm(y - app), infl_p))

    form_parameters[i] = h[0][1]
print(100, '%')

A = f.convex_constraints_matrix(form_parameters)
APPROXIMATION = f.numeric_sol(y, A)

infl_p = form_parameters[1:-1]
periods_1 = [infl_p[i+2] - infl_p[i] for i in range(len(infl_p)-2)]
periods = np.array([(periods_1[i] + periods_1[i+1])/2 for i in range(0, len(periods_1)-1, 2)])
avg_period = periods.mean()

# graphs
fig, ax1 = plt.subplots(1)
ax1.plot(x, y, 'ro', markersize=2, label='experimental data')
ax1.plot(x, APPROXIMATION, 'b', linewidth=1, label='result of morphological filtering')
ax1.plot(x[infl_p], APPROXIMATION[infl_p], 'go', markersize=5, label='inflection points')
ax1.set_xlabel("Time (1 tick = 30 min)")
ax1.set_ylabel("CO2 concentration")
legend1 = ax1.legend(loc='best', shadow=True, fontsize='large')

plt.show()

'''
# ax2.plot(
#     x, y-APPROXIMATION, 'g', linewidth=1, label="residual series"
#     )
# ax2.set_xlabel("Time (1 tick = 30 min)")
# ax2.set_ylabel("CO2 concentration")
# legend2 = ax2.legend(loc='best', shadow=True, fontsize='large')
'''
