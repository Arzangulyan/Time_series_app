"""
functions for the approximation
"""
import numpy as np


def checkconvexup(q):

    for i in range(1, len(q)-1):
        if (2*q[i] < q[i-1]+q[i+1]) and (abs(2*q[i]-q[i-1]-q[i+1]) > (1e-10)):
            return False
    return True


def checkconvexdown(q):

    q = np.negative(q)
    return checkconvexup(q)


def mono_incr(q):
    if (q[0:-1] > q[1:]).any():
        return False
    else:
        return True


def mono_decr(q):
    if (q[0:-1] < q[1:]).any():
        return False
    else:
        return True


def leastsq(f):
    n = len(f)
    x = np.linspace(1, n, n)
    A = np.vstack([x, np.ones(n)]).T
    a, b = np.linalg.lstsq(A, f, rcond=None)[0]

    return np.array(a*x + b)


def mono_incr_leastsq(f):
    n = len(f)
    x = np.linspace(1, n, n)
    A = np.vstack([x, np.ones(n)]).T
    a, b = np.linalg.lstsq(A, f, rcond=None)[0]
    if a >= 0:
        return np.array(a*x + b)
    else:
        popt = np.polyfit(x, f, 0)
        return np.polyval(popt, x)


def mono_decr_leastsq(f):
    n = len(f)
    x = np.linspace(1, n, n)
    A = np.vstack([x, np.ones(n)]).T
    a, b = np.linalg.lstsq(A, f, rcond=None)[0]
    if a <= 0:
        return np.array(a*x + b)
    else:
        popt = np.polyfit(x, f, 0)
        return np.polyval(popt, x)


def projconvup(f, init, final, points):
    curr_segment = f[init:final]
    segment_len = len(curr_segment)
    if points[init][final] != 0 or segment_len <= 3:
        return build_proj_conv_up(f, init, final, points)
    else:
        for i in range(1, segment_len):
            if points[init][init+i] != 0 or i <= 3:
                leftproj = build_proj_conv_up(f, init, init+i, points)
            else:
                leftproj = projconvup(f, init, init+i, points)

            if segment_len - i <= 3 or points[init+i][final] != 0:
                rightproj = build_proj_conv_up(f, init+i, final, points)
            else:
                rightproj = projconvup(f, init+i, final, points)

            q = np.append(leftproj, rightproj)
            if (checkconvexup(q)):
                points[init][final] = init + i
                return q

        points[init][final] = -1
        return leastsq(curr_segment)


def projconvdown(f, init, final, points):
    curr_segment = f[init:final]
    segment_len = len(curr_segment)
    if points[init][final] != 0 or segment_len <= 3:
        return build_proj_conv_up(f, init, final, points)
    else:
        for i in range(1, segment_len):
            if i <= 3 or points[init][init+i] != 0:
                leftproj = build_proj_conv_down(f, init, init+i, points)
            else:
                leftproj = projconvdown(f, init, init+i, points)

            if segment_len - i <= 3 or points[init+i][final] != 0:
                rightproj = build_proj_conv_down(f, init+i, final, points)
            else:
                rightproj = projconvdown(f, init+i, final, points)

            q = np.append(leftproj, rightproj)
            if (checkconvexdown(q)):
                points[init][final] = init + i
                return q

        points[init][final] = -1
        return leastsq(curr_segment)


def proj_mono_incr(f, init, final, points):
    curr_segment = f[init:final]
    segment_len = len(curr_segment)
    if points[init][final] != 0 or segment_len <= 3:
        return build_proj_mono_incr(f, init, final, points)
    else:
        for i in range(1, segment_len):
            if i <= 3 or points[init][init+i] != 0:
                leftproj = build_proj_mono_incr(f, init, init+i, points)
            else:
                leftproj = proj_mono_incr(f, init, init+i, points)

            if segment_len - i <= 3 or points[init+i][final] != 0:
                rightproj = build_proj_mono_incr(f, init+i, final, points)
            else:
                rightproj = proj_mono_incr(f, init+i, final, points)

            q = np.append(leftproj, rightproj)
            if (mono_incr(q)):
                points[init][final] = init + i
                return q

        points[init][final] = -1
        return mono_incr_leastsq(curr_segment)


def proj_mono_decr(f, init, final, points):
    curr_segment = f[init:final]
    segment_len = len(curr_segment)
    if points[init][final] != 0 or segment_len <= 3:
        return build_proj_mono_decr(f, init, final, points)
    else:
        for i in range(1, segment_len):
            if i <= 3 or points[init][init+i] != 0:
                leftproj = build_proj_mono_decr(f, init, init+i, points)
            else:
                leftproj = proj_mono_decr(f, init, init+i, points)

            if segment_len - i <= 3 or points[init+i][final] != 0:
                rightproj = build_proj_mono_decr(f, init+i, final, points)
            else:
                rightproj = proj_mono_decr(f, init+i, final, points)

            q = np.append(leftproj, rightproj)
            if (mono_decr(q)):
                points[init][final] = init + i
                return q

        points[init][final] = -1
        return mono_decr_leastsq(curr_segment)


def is_ext(f, ind):
    if len(f) < 3:
        return False
    elif (f[ind-1] < f[ind] and f[ind+1] < f[ind]) or (f[ind-1] > f[ind] and f[ind+1] > f[ind]):
        return True
    else:
        return False


def is_max(f, ind):
    if ind == 0 or ind == len(f)-1:
        return False
    elif (f[ind-1] < f[ind] and f[ind+1] < f[ind]):
        return True
    else:
        return False


def is_min(f, ind):
    if ind == 0 or ind == len(f)-1:
        return False
    elif (f[ind-1] > f[ind] and f[ind+1] > f[ind]):
        return True
    else:
        return False


def build_proj_conv_up(y, init, final, points):
    if (points[init][final] == -1):
        return leastsq(y[init:final])
    elif (final - init == 1) or (final - init == 2):
        return y[init:final]
    elif (final - init == 3):
        if checkconvexup(y[init:final]):
            return y[init:final]
        else:
            return leastsq(y[init:final])
    else:
        return np.append(build_proj_conv_up(y, init, points[init][final], points),
                         build_proj_conv_up(y, points[init][final], final, points))


def build_proj_conv_down(y, init, final, points):
    if (points[init][final] == -1):
        return leastsq(y[init:final])
    elif (final - init == 1) or (final - init == 2):
        return y[init:final]
    elif (final - init == 3):
        if checkconvexdown(y[init:final]):
            return y[init:final]
        else:
            return leastsq(y[init:final])
    else:
        return np.append(build_proj_conv_down(y, init, points[init][final], points),
                         build_proj_conv_down(y, points[init][final], final, points))


def build_proj_mono_incr(y, init, final, points):
    if (points[init][final] == -1):
        return mono_incr_leastsq(y[init:final])
    elif (final - init == 1) or (final - init == 2):
        return y[init:final]
    elif (final - init == 3):
        if mono_incr(y[init:final]):
            return y[init:final]
        else:
            return mono_incr_leastsq(y[init:final])
    else:
        return np.append(build_proj_mono_incr(y, init, points[init][final], points),
                         build_proj_mono_incr(y, points[init][final], final, points))


def build_proj_mono_decr(y, init, final, points):
    if (points[init][final] == -1):
        return mono_decr_leastsq(y[init:final])
    elif (final - init == 1) or (final - init == 2):
        return y[init:final]
    elif (final - init == 3):
        if mono_decr(y[init:final]):
            return y[init:final]
        else:
            return mono_decr_leastsq(y[init:final])
    else:
        return np.append(build_proj_mono_decr(y, init, points[init][final], points),
                         build_proj_mono_decr(y, points[init][final], final, points))


def dif(q, f):
    return (sum(i*i for i in (q-f)))


def find_closest(array, value, curr_index):
    for i in range(curr_index+1, len(array)):
        if array[i][0] == value:
            return i
    return 'err'


def find_next(evolution, i):
    for j in range(i+1, len(evolution)-1):
        if evolution[j] != -1:
            return j
    return None


def find_min_delts(delts):
    min_ind = delts[0][0]
    min_val = delts[0][1]
    for i in range(1, len(delts)):
        if delts[i][1] < min_val:
            min_ind = delts[i][0]
            min_val = delts[i][1]
    return min_ind


def optimal_point(delts, priority):
    if priority == 1:
        if delts[0]:
            return find_min_delts(delts[0])
        else:
            return find_min_delts(delts[1])
    elif priority == 0:
        if delts[0] and delts[1]:
            return find_min_delts(delts[0]+delts[1])
        elif delts[0]:
            return find_min_delts(delts[0])
        else:
            return find_min_delts(delts[1])
