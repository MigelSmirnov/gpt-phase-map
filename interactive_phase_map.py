from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource, CDSView, BooleanFilter, HoverTool, TapTool,
    Legend, LegendItem, Div, Select, CustomJS, LabelSet
)
from bokeh.layouts import column as bokeh_column, row as bokeh_row
import pandas as pd
import numpy as np
import json
from collections import defaultdict

# ====== CONFIG ======
CSV_PATH = "phase_points.csv"
REGIME_JSON = "hash_to_text.json"
LOGS_JSON = "logs.json"

degree_levels = [0, 1, 2, 5, 10, 15, 20, 30, 40, 60]
radii_vals = np.linspace(0.2, 1.0, len(degree_levels))
radius_map = dict(zip(degree_levels, radii_vals))


def get_radius(deg):
    val = abs(float(deg))
    closest = min(degree_levels, key=lambda x: abs(x - val))
    return radius_map[closest]


# ====== LOAD DATA ======
df = pd.read_csv(CSV_PATH)

dirs = sorted(df["dir"].unique())
plane_angles = np.linspace(0, np.pi, len(dirs)+2)[1:-1]
axis_map = {d: plane_angles[i] for i, d in enumerate(dirs)}

def point_xy(dir_i, angle_deg):
    theta = axis_map.get(dir_i)
    if theta is None:
        return None, None
    theta_plot = theta if angle_deg >= 0 else theta + np.pi
    r = get_radius(abs(angle_deg))
    return r * np.cos(theta_plot), r * np.sin(theta_plot)


# ====== TOP-4 COLORS ======
top = df["regime"].value_counts().head(4).index.tolist()
COLOR = {
    top[0]: "black",
    top[1]: "red",
    top[2]: "blue",
    top[3]: "green",
}
df["color"] = df["regime"].map(COLOR).fillna("lightgray")

# ====== XY ======
xs, ys = [], []
for _, r in df.iterrows():
    x, y = point_xy(int(r.dir), float(r.angle))
    xs.append(x)
    ys.append(y)
df["x"] = xs
df["y"] = ys

# ====== TEXT ======
with open(REGIME_JSON, "r", encoding="utf-8") as f:
    hash_to_text = json.load(f)

df["text"] = df["regime"].map(hash_to_text).fillna("(нет текста)")
df["size"] = 6

source = ColumnDataSource(df)

active_filter = BooleanFilter([True] * len(df))
inactive_filter = BooleanFilter([False] * len(df))

active_view = CDSView(filter=active_filter)
inactive_view = CDSView(filter=inactive_filter)


# ====== FIGURE ======
output_file("phase_map_fast.html")

p = figure(
    width=700, height=700,
    match_aspect=True,
    tools="pan,wheel_zoom,reset",
    title="Interactive GPT Phase Map",
    output_backend="webgl"
)
p.grid.visible = False


# =============================
#   🔥 ОКРУЖНОСТИ = ОДИН multi_line
# =============================
circle_xs = []
circle_ys = []

for r in radii_vals:
    if r > 0:
        t = np.linspace(0, 2*np.pi, 200)
        circle_xs.append((r*np.cos(t)).tolist())
        circle_ys.append((r*np.sin(t)).tolist())

p.multi_line(circle_xs, circle_ys,
             color="gray",
             line_alpha=0.15)


# =============================
#   🔥 ПОДПИСИ ГРАДУСОВ = ОДИН LabelSet
# =============================
label_source = ColumnDataSource(dict(
    x=[r+0.01 for r in radii_vals if r > 0],
    y=[0 for _ in radii_vals if _ > 0],
    text=[f"{deg}°" for deg in degree_levels if deg > 0],
))

labels = LabelSet(
    x='x', y='y', text='text',
    source=label_source,
    text_font_size='8pt',
    text_alpha=0.5
)

p.add_layout(labels)


# =============================
#   ЛИНИИ МЕЖДУ ТОЧКАМИ
# =============================
by_dir_pos = defaultdict(list)
by_dir_neg = defaultdict(list)

for _, row in df.iterrows():
    d = int(row.dir)
    ang = float(row.angle)
    regime = row.regime
    x, y = row.x, row.y
    
    if x is None:
        continue
    
    r = get_radius(abs(ang))
    
    if ang >= 0:
        by_dir_pos[d].append((r, x, y, regime))
    else:
        by_dir_neg[d].append((r, x, y, regime))

def color_for(regime):
    return COLOR.get(regime, "lightgray")

lines_by_color = defaultdict(lambda: {'xs': [], 'ys': []})

for d, pts in by_dir_pos.items():
    pts = sorted(pts, key=lambda t: t[0])
    for (r1,x1,y1,_), (r2,x2,y2,reg2) in zip(pts[:-1], pts[1:]):
        col = color_for(reg2)
        lines_by_color[col]['xs'].append([x1, x2])
        lines_by_color[col]['ys'].append([y1, y2])

for d, pts in by_dir_neg.items():
    pts = sorted(pts, key=lambda t: t[0])
    for (r1,x1,y1,_), (r2,x2,y2,reg2) in zip(pts[:-1], pts[1:]):
        col = color_for(reg2)
        lines_by_color[col]['xs'].append([x1, x2])
        lines_by_color[col]['ys'].append([y1, y2])

for col, lines in lines_by_color.items():
    p.multi_line(lines['xs'], lines['ys'], color=col, line_width=1.2, alpha=0.8)


# =============================
#   ТОЧКИ
# =============================
p.scatter(
    x='x', y='y',
    source=source,
    view=inactive_view,
    size='size',
    fill_color='color',
    fill_alpha=0.12,
    line_color=None,
)

r_active = p.scatter(
    x='x', y='y',
    source=source,
    view=active_view,
    size='size',
    fill_color='color',
    fill_alpha=0.95,
    line_color=None,
)

p.add_tools(HoverTool(renderers=[r_active], mode="mouse"))
p.add_tools(TapTool())


# =============================
#   ЛЕГЕНДА
# =============================
legend_items = []
labels = ["BASE", "R", "B", "G"]
colors = ["black", "red", "blue", "green"]

for label, color, regime_hash in zip(labels, colors, top):
    dummy = p.scatter([], [], size=8, color=color, alpha=0.9)
    legend_items.append(LegendItem(label=f"{label}: {regime_hash[:8]}", renderers=[dummy]))

legend = Legend(items=legend_items, location="top_right")
p.add_layout(legend)


# =============================
#   UI
# =============================
text_div = Div(text="<b>Кликни по точке</b>", width=600)

plane_select = Select(
    title="Плоскость (dir):",
    value="ALL",
    options=["ALL"] + sorted(df["dir"].astype(str).unique().tolist())
)


# ====== ЛОГИ ДЛЯ ЭНТРОПИИ ======
with open(LOGS_JSON, "r", encoding="utf-8") as f:
    logs_data = json.load(f)

logs_json = {str(k): v for k, v in logs_data.items()}

entropy_source = ColumnDataSource(data=dict(
    angle=[], entropy=[], regime=[], color=[], text=[]
))


# ====== JS CALLBACK ======
callback = CustomJS(
    args=dict(
        source=source,
        div=text_div,
        select=plane_select,
        entropy_source=entropy_source,
        logs_data=logs_json,
        color_map=COLOR
    ),
    code=""" ... тот же код что был ... """
)

source.selected.js_on_change("indices", callback)


layout = bokeh_row(
    p,
    bokeh_column(plane_select, text_div)
)

show(layout)
