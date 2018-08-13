import math

from structural import System
import matplotlib.animation as anim
import matplotlib
import matplotlib.pyplot as plt

# Units: kN / m

# Create a new system object.
EA = 1e12
EI_0 = 7e6

# Druk op voorhar
Rv = 100  # kN

# Verdeelde belasting
wcw1 = 23 * 15
wcw2 = 23 * 15
wa = 2 * 15
wb = 2 * 15

life = 100 * 365 * 24
month = 365 * 24 / 12
month = 365 * 2


def create_system(EIt):
    sys = System()
    # Add beams to the system. Positive z-axis is down, positive x-axis is the right.
    cw1 = sys.add_element(coordinates=[[-5.5, 0], [-3.5, 0]], EA=EA, EI=EIt)
    cw2 = sys.add_element(coordinates=[[-3.5, 0], [-1.5, 0]], EA=EA, EI=EIt)
    a = sys.add_element(coordinates=[[-1.5, 0], [0, 0]], EA=EA, EI=EIt)
    b = sys.add_element(coordinates=[[0, 0], [18, 0]], EA=EA, EI=EIt)

    sys.roll_support(node=cw2.node_1)
    sys.hinged_support(node=a.node_2)

    sys.q_load(q=wcw1, element=cw1, direction='SE')
    sys.q_load(q=wcw2, element=cw2, direction='SE')
    sys.q_load(q=wa, element=a, direction='SE')
    sys.q_load(q=wb, element=b, direction='SE')
    sys.point_load(Fz=-Rv, node=b.node_2)
    return sys


def EI(t):
    p = 0.75
    E_0 = 32.3
    E_UD = 39.3
    n = 0.01
    r = E_0 / (p * E_UD)
    if t == 0:
        return EI_0
    return EI_0 / (r * t ** n)


def life(yrs, step_hrs):
    l = yrs * 365 * 24
    t = 0
    while t < l:
        yield t
        t += step_hrs


def creep(yrs, step_hrs):
    for t in life(yrs, step_hrs):
        sys = create_system(EI(t))
        sys.solve()
        yield sys, t


FFMpegWriter = anim.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

# Initial step
system = create_system(EI_0)
system.solve()
plot = system.plot(structure=True)
plot.displacement_factor = 2 * plot.displacement_factor
system.plot(displacements=True, plot=plot, rot=math.radians(8.531), center=(0, 0))
# plot.figure.show()
# plt.show()

def rot_angle(plot):
    uz = plot.system.elements[-1].node_2.uz
    l = 18
    theta = math.atan(plot.displacement_factor * uz / l)
    return theta

# print(l, ux, uz, plot.displacement_factor)

with writer.saving(plot.figure, "writer_test.mp4", 300):
    for s, t in creep(1 / 12, step_hrs=1):
        plot.system = s

        s.plot(structure=True, displacements=True, plot=plot)
        plot.clear()
        theta = rot_angle(plot)
        print(t / 24, math.degrees(theta))
        s.plot(structure=True, displacements=True, plot=plot, rot=theta)
        writer.grab_frame()
        # plot.figure.show()
        plot.clear()

# Add supports.
# system.fix_support(node=e1.node_1)
# Add a rotational spring at node 4.
# system.spring_support(node=4, translation='rot', K=4000)

# Add loads.
# system.point_load(Fx=30, node=e1.node_2)
# system.q_load(q=10, element=e1, direction='NW')  # -1 = up / left, 1 = down / right
"""
system.solve()



    for i in range(100):
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        l.set_data(x0, y0)
        writer.grab_frame()

p = system.plot(structure=True, displacements=True)
p.figure.show()
time.sleep(5)

# system.plot_reaction_force()
# system.plot_normal_force()
# system.plot_shear_force()
# system.plot_bending_moment()
# system.plot_displacement()
"""