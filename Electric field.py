import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider

# Grid settings
x_min, x_max, y_min, y_max = -5.0, 5.0, -5.0, 5.0
n = 40  # resolution for quiver grid
n_stream = 100  # resolution for streamplot grid

# Initial charge properties
q_sign = 1.0   # +1 or -1 (sign from radio)
q_mag = 1.0    # adjustable magnitude
charge_pos = []  # no charge initially
charges = []
# Prepare grid
x = np.linspace(x_min, x_max, n)
y = np.linspace(y_min, y_max, n)
X, Y = np.meshgrid(x, y)

xs = np.linspace(x_min, x_max, n_stream)
ys = np.linspace(y_min, y_max, n_stream)
Xs, Ys = np.meshgrid(xs, ys)

def compute_field(q_sign,q_mag,pos, X, Y, eps=1e-3):
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    for i,p in enumerate(pos):
        q = q_sign[i]*q_mag
        dx = X - p[0]
        dy = Y - p[1]
        r2 = dx**2 + dy**2 + eps
        r = np.sqrt(r2)
        Ex += q * dx / (r2 * r)
        Ey += q * dy / (r2 * r)
    return Ex, Ey

# Create figure and axes
plt.close('all')
fig = plt.figure(figsize=(9,6))
ax = fig.add_axes([0.05, 0.05, 0.65, 0.9])
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal', 'box')
ax.set_title('Electric field of a point charge\nClick to place charge. Use controls to adjust.')

# Widgets
rax = fig.add_axes([0.72, 0.7, 0.22, 0.2])
radio = RadioButtons(rax, ('Positive', 'Negative'), active=0)

bax = fig.add_axes([0.72, 0.62, 0.22, 0.06])
clear_button = Button(bax, 'Clear charge')

sax = fig.add_axes([0.72, 0.55, 0.22, 0.03])
slider = Slider(sax, 'Magnitude', 0.1, 5.0, valinit=1.0)

info_ax = fig.add_axes([0.72, 0.05, 0.25, 0.45])
info_ax.axis('off')
info_text = info_ax.text(0, 0.95, 'Instructions:\n\n'
    '- Click in the square to place the charge.\n'
    '- Radio sets charge sign.\n'
    '- Slider sets magnitude.\n'
    '- Clear removes charge.\n',
    va='top', wrap=True)

# References to artists
Q = None
S = None
charge_artist = []
coord_text = ax.text(0.02, 0.98, 'Charge: None', transform=ax.transAxes,
                     va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

def update_plot():
    global Q, S, charge_artist, coord_text
    # remove previous artists
    if Q is not None:
        Q.remove()
        Q = None

    #if S is not None:
    #    S.lines.remove()
    #    S.arrows.remove()
    #    S = None


    if len(charge_artist)>0:
        for art in charge_artist:
            art.remove()
        charge_artist = []

    if len(charge_pos)<1:
        # draw empty
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)
        Q = ax.quiver(X, Y, Ex, Ey, pivot='mid', scale=1)
        coord_text.set_text('Charge: None')
        
    else:
        #q = q_sign * q_mag
        Ex, Ey = compute_field(charges,q_mag, charge_pos, X, Y)
        mag = np.sqrt(Ex**2 + Ey**2)
        mag_norm = np.where(mag==0, 1, mag)
        U, V = Ex/mag_norm, Ey/mag_norm
        Q = ax.quiver(X, Y, U, V, mag, cmap='viridis', pivot='mid', scale=40)

        ## Field lines (streamplot on finer grid)
        #Exs, Eys = compute_field(q, charge_pos, Xs, Ys)
        #speed = np.sqrt(Exs**2 + Eys**2)
        #S = ax.streamplot(xs, ys, Exs, Eys, color=speed, linewidth=1, cmap='plasma', density=1.2)

        for i,p in enumerate(charge_pos):
            charge_artist += [ax.scatter(p[0], p[1], s=50,
                                       c=('red' if charges[i]>0 else 'blue'), edgecolors='k', zorder=3)]

        #coord_text.set_text(f'Charge: {"+" if q>0 else "-"}{abs(q):.2f} at ({charge_pos[0]:.2f}, {charge_pos[1]:.2f})')

    fig.canvas.draw_idle()

def on_radio(label):
    global q_sign
    q_sign = 1.0 if label == 'Positive' else -1.0
    update_plot()

def on_click(event):
    global charge_pos,q_sign,charges
    if event.inaxes == ax and event.button == 1:
        charge_pos += [np.array([event.xdata, event.ydata])]
        charges +=[q_sign]
        update_plot()

def on_clear(event):
    global charge_pos,charges
    charge_pos = []
    charges = []
    #charge_artist = None
    update_plot()

def on_slider(val):
    global q_mag
    q_mag = slider.val
    update_plot()

# Connect
radio.on_clicked(on_radio)
fig.canvas.mpl_connect('button_press_event', on_click)
clear_button.on_clicked(on_clear)
slider.on_changed(on_slider)

# Initialize
update_plot()
plt.show()
