# ===== Standard Library =====
import json
from collections import defaultdict
from pathlib import Path

# ===== Third-party =====
import numpy as np
import pandas as pd

# ===== Bokeh =====
from bokeh.plotting import figure, show, output_file
from bokeh.io import curdoc
from bokeh.events import DocumentReady

from bokeh.models import (
    ColumnDataSource,
    CDSView,
    BooleanFilter,
    HoverTool,
    TapTool,
    Legend,
    LegendItem,
    Div,
    Select,
    CustomJS,
    LabelSet,
    LinearAxis,
    Range1d,
)

from bokeh.layouts import column as bokeh_column, row as bokeh_row

# ====== CONFIG ======
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CSV_PATH = DATA_DIR / "points_dataset.csv"
REGIME_JSON = DATA_DIR / "regime_text_map.json"

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
plane_angles = np.linspace(0, np.pi, len(dirs) + 2)[1:-1]
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

# ====== TEXT (из JSON) ======
with open(REGIME_JSON, "r", encoding="utf-8") as f:
    hash_to_text = json.load(f)

df["text"] = df["regime"].map(hash_to_text).fillna("(нет текста)")
df["size"] = 6

# === ВАЖНО ===
df = df[df["x"].notna() & df["y"].notna()].copy()
df["text"] = df["text"].fillna("(нет текста)").astype(str)

print("DEBUG LENGTHS:",
      len(df["x"]),
      len(df["y"]),
      len(df["text"]))

# ====== SOURCE ======
source = ColumnDataSource(df)

# ===== SECOND SOURCE: entropy trajectory =====
entropy_source = ColumnDataSource(dict(
    angle=[],
    xpos=[],
    entropy=[],
    color=[],
    text=[]
))

active_filter = BooleanFilter([True] * len(df))
inactive_filter = BooleanFilter([False] * len(df))

active_view = CDSView(filter=active_filter)
inactive_view = CDSView(filter=inactive_filter)

# ====== FIGURE ======
output_file("phase_map_fast.html")

p = figure(
    width=700, height=700,
    match_aspect=True,
    tools="pan,wheel_zoom,reset,tap",
    active_drag=None,
    active_scroll=None,
    active_tap="auto",
    title="Interactive GPT Phase Map",
    output_backend="webgl"
)

p.grid.visible = False

# =============================
#   ОКРУЖНОСТИ = ОДИН multi_line
# =============================
circle_xs = []
circle_ys = []

for r in radii_vals:
    if r > 0:
        t = np.linspace(0, 2 * np.pi, 200)
        circle_xs.append((r * np.cos(t)).tolist())
        circle_ys.append((r * np.sin(t)).tolist())

p.multi_line(circle_xs, circle_ys,
             color="gray",
             line_alpha=0.15)

# =============================
#   ПОДПИСИ ГРАДУСОВ = ОДИН LabelSet
# =============================

deg_pos = [d for d in degree_levels if d > 0]
rad_pos = [radius_map[d] for d in deg_pos]

label_source = ColumnDataSource(dict(
    x=[r + 0.01 for r in rad_pos],
    y=[0 for _ in rad_pos],
    text=[f"{d}°" for d in deg_pos],
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
    for (r1, x1, y1, _), (r2, x2, y2, reg2) in zip(pts[:-1], pts[1:]):
        col = color_for(reg2)
        lines_by_color[col]['xs'].append([x1, x2])
        lines_by_color[col]['ys'].append([y1, y2])

for d, pts in by_dir_neg.items():
    pts = sorted(pts, key=lambda t: t[0])
    for (r1, x1, y1, _), (r2, x2, y2, reg2) in zip(pts[:-1], pts[1:]):
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

hover = HoverTool(
    renderers=[r_active],
    tooltips=[
        ("Угол", "@angle°"),
        ("Режим", "@regime"),
        ("Текст", "@text{safe}"),
    ],
    mode="mouse"
)
p.add_tools(hover)

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


# PNG для мобильной версии
png_div = Div(
    text='',
    width=700
)

# список значений энтропии
entropy_list = Div(text="", width=600)

# ====== JS CALLBACK (клик по точке) ======
callback = CustomJS(
    args=dict(
        source=source,
        entropy_source=entropy_source,
        div=text_div
    ),
    code="""
    const inds = source.selected.indices;
    if (inds.length === 0) return;

    const i = inds[0];
    const data = source.data;

    const target_dir = data['dir'][i].toString();
    const target_regime = data['regime'][i];
    
    // Обновляем текст справа
    const txt = data['text'][i] || "(нет текста)";
    const html_txt = txt.replace(/\\n/g, "<br>");
    
    div.text = `
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;">
            <b>Направление (dir):</b> ${target_dir}<br>
            <b>Угол:</b> ${data['angle'][i]}°<br>
            <b>Режим:</b> <span style="color:${data['color'][i]}; font-weight: bold;">${target_regime}</span><br>
            <hr>
            <b>Текст:</b><br>
            <div style="font-family: monospace; font-size: 11px;">${html_txt}</div>
        </div>
    `;

    const new_data = {
        angle: [],
        entropy: [],
        color: [],
        text: []
    };

    // Собираем данные по той же dir
    for (let k = 0; k < data['dir'].length; k++) {
        if (data['dir'][k].toString() === target_dir) {
            new_data.angle.push(data['angle'][k]);
            new_data.entropy.push(data['entropy'][k]);
            new_data.text.push(data['text'][k]);
            
            if (data['regime'][k] === target_regime) {
                new_data.color.push(data['color'][k]);
            } else {
                new_data.color.push("#E6E6E6");
            }
        }
    }

    // Сортировка по углу

    const angles_sorted = Array.from(new_data.angle.keys())
        .sort((a,b)=> new_data.angle[a]-new_data.angle[b])
        .map(i => new_data.angle[i]);

    // равномерные X с нулём по центру
    const N = angles_sorted.length;
    const xpos = angles_sorted.map((a, j) => j - (N-1)/2);

    entropy_source.data = {
        angle: angles_sorted,
        xpos: xpos,
        entropy: angles_sorted.map(a =>
            new_data.entropy[new_data.angle.indexOf(a)]
        ),
        color: angles_sorted.map(a =>
            new_data.color[new_data.angle.indexOf(a)]
        ),
        text: angles_sorted.map(a =>
            new_data.text[new_data.angle.indexOf(a)]
        )
    };

    entropy_source.change.emit();
    
    let s = "";
    for(let j=0;j<entropy_source.data.angle.length;j++){
      s += `${entropy_source.data.angle[j]} = ${entropy_source.data.entropy[j].toFixed(4)}<br>`;
    }
    entropy_list.text = s;


    """
)

source.selected.js_on_change("indices", callback)

# =============================
#   ПРАВЫЙ ГРАФИК — ENTROPY
# =============================
p2 = figure(
    width=600, height=400,
    title="Entropy trajectory",
    x_axis_label="Angle (°)", y_axis_label="Entropy",
    tools="pan,wheel_zoom,reset"
)

p2.line('xpos', 'entropy', source=entropy_source, color="gray", line_width=1, alpha=0.3)

r2 = p2.scatter(
    'xpos', 'entropy',
    source=entropy_source,
    size=10,
    color='color',
    line_color="black",
    line_width=0.5
)

p2.add_tools(HoverTool(
    renderers=[r2],
    tooltips=[
        ("Угол", "@angle°"),
        ("Энтропия", "@entropy{0.000}"),
        ("Режим", "@text")
    ]
))

layout = bokeh_row(
    bokeh_column(
        png_div,   # ← сверху PNG (мобильная версия)
        p          # ← ниже интерактивная карта (десктоп)
    ),
    bokeh_column(
        plane_select,
        text_div,
        p2,
        entropy_list   # ← список значений энтропии
    )
)

# ====== CALLBACK ДЛЯ ВЫБОРА ПЛОСКОСТИ (SELECT) ======
select_callback = CustomJS(
    args=dict(
        source=source,
        entropy_source=entropy_source,
        div=text_div,
        active_filter=active_filter,
        inactive_filter=inactive_filter
    ),
    code="""
    const val = cb_obj.value;
    const data = source.data;
    const n = data['dir'].length;

    // 1. Маски для активных/неактивных точек
    const act = [];
    const inact = [];

    for (let i = 0; i < n; i++) {
        if (val === "ALL" || data['dir'][i].toString() === val) {
            act.push(true);
            inact.push(false);
        } else {
            act.push(false);
            inact.push(true);
        }
    }

    active_filter.booleans = act;
    inactive_filter.booleans = inact;
    source.change.emit();

    // 2. График энтропии
    if (val !== "ALL") {

        const new_data = { angle: [], entropy: [], color: [], text: [] };

        for (let k = 0; k < n; k++) {
            if (data['dir'][k].toString() === val) {
                new_data.angle.push(data['angle'][k]);
                new_data.entropy.push(data['entropy'][k]);
                new_data.text.push(data['text'][k]);
                new_data.color.push("#888888");
            }
        }

        // сортировка по углу
        const order = Array.from(new_data.angle.keys());
        order.sort((a, b) => new_data.angle[a] - new_data.angle[b]);

        const angles_sorted = order.map(i => new_data.angle[i]);

        // равномерный шаг + 0 в центре
        const N = angles_sorted.length;
        const xpos = angles_sorted.map((a, j) => j - (N - 1) / 2);

        entropy_source.data = {
            angle: angles_sorted,
            xpos: xpos,
            entropy: order.map(i => new_data.entropy[i]),
            color: order.map(i => new_data.color[i]),
            text: order.map(i => new_data.text[i])
        };

    } else {

        entropy_source.data = { angle: [], xpos: [], entropy: [], color: [], text: [] };
    }

    entropy_source.change.emit();
    
    let s = "";
    for(let j=0;j<entropy_source.data.angle.length;j++){
      s += `${entropy_source.data.angle[j]} = ${entropy_source.data.entropy[j].toFixed(4)}<br>`;
    }
    entropy_list.text = s;



    """
)

plane_select.js_on_change("value", select_callback)



mobile_js = CustomJS(args=dict(
    p=p,
    p2=p2,
    png_div=png_div
), code="""
const is_mobile = window.innerWidth < 820;

if(is_mobile){

    // Спрятать интерактивную карту
    p.visible = false;

    // Показать PNG вместо неё
    png_div.text = `
        <img src="phase_map_static.png"
             style="width:100%; max-width:900px; border-radius:10px;">
    `;

    // Убрать hover с графика энтропии
    p2.tools = p2.tools.filter(t => t.type !== 'hover');
}
""")


curdoc().js_on_event(DocumentReady, mobile_js)


show(layout)


