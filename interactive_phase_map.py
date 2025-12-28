import pandas as pd
import numpy as np
import json
import re
import os
import sys

from collections import defaultdict
from bokeh.models import BooleanFilter, CDSView, Circle as BokehCircle
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, Div, Select, CustomJS, Legend, LegendItem
from bokeh.layouts import row as bokeh_row, column as bokeh_column
from bokeh.io import output_file


# ====== ПУТИ ======
import os
import sys

if '__file__' in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

CSV_PATH = os.path.join(BASE_DIR, "data", "points_dataset.csv")
REGIME_JSON = os.path.join(BASE_DIR, "data", "regime_text_map.json")
LOG_PATH = os.path.join(BASE_DIR, "data", "phase_rotation_log_2025-12-28_01-13-28.txt")

# ====== ПАРСИНГ ЛОГОВ ======
def parse_logs(log_path):
    """Парсит логи и возвращает dict: {dir: [{angle, entropy, regime, text}, ...]}"""
    data_by_dir = defaultdict(list)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разбиваем по направлениям
    dir_blocks = re.split(r'====== DIRECTION #(\d+) ======', content)
    
    for i in range(1, len(dir_blocks), 2):
        dir_num = int(dir_blocks[i])
        block = dir_blocks[i+1]
        
        # Ищем все записи с углами (пропускаем первый +60)
        pattern = r'----- DIR \d+\s+ANG ([+-]?\d+\.?\d*)° -----\s+cos\(rot,target\)=\s*([\d.]+)\s+entropy\(first\)=\s*([\d.]+)\s+top1=.*?p=([\d.]+)\s+regime=\s*(\w+)\s+text\(first line\)=\s*(.+?)(?=\n----|$)'
        
        matches = re.findall(pattern, block, re.DOTALL)
        
        # Пропускаем первую запись если это +60
        start_idx = 1 if matches and float(matches[0][0]) == 60.0 else 0
        
        for match in matches[start_idx:]:
            angle = float(match[0])
            cos_val = float(match[1])
            entropy = float(match[2])
            prob = float(match[3])
            regime = match[4]
            text = match[5].strip()
            
            data_by_dir[dir_num].append({
                'angle': angle,
                'cos': cos_val,
                'entropy': entropy,
                'prob': prob,
                'regime': regime,
                'text': text
            })
    
    return data_by_dir

print("Парсинг логов...")
logs_data = parse_logs(LOG_PATH)
print(f"Загружено данных для {len(logs_data)} направлений")


# ====== РАДИАЛЬНАЯ ШКАЛА ======
degree_levels = [0,1,2,5,10,15,20,30,40,60]
R_max = 1.0
t_radii = np.linspace(0.05, 1, len(degree_levels))
radii_vals = R_max * np.sqrt(t_radii)
radius_map = {d: r for d, r in zip(degree_levels, radii_vals)}

def get_radius(deg):
    val = abs(float(deg))
    closest = min(degree_levels, key=lambda x: abs(x - val))
    return radius_map[closest]


# ====== ЧИТАЕМ CSV ======
df = pd.read_csv(CSV_PATH)

# ====== ОСИ ======
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


# ====== ТОП-4 РЕЖИМА ======
top = df["regime"].value_counts().head(4).index.tolist()
COLOR = {
    top[0]: "black",
    top[1]: "red",
    top[2]: "blue",
    top[3]: "green",
}
df["color"] = df["regime"].map(COLOR).fillna("lightgray")


# ====== КООРДИНАТЫ ======
xs, ys = [], []
for _, r in df.iterrows():
    x, y = point_xy(int(r.dir), float(r.angle))
    xs.append(x)
    ys.append(y)
df["x"] = xs
df["y"] = ys


# ====== Bokeh-источник ======
with open(REGIME_JSON, "r", encoding="utf-8") as f:
    hash_to_text = json.load(f)

df["text"] = df["regime"].map(hash_to_text).fillna("(нет текста)")
df["alpha"] = 0.9
df["size"] = 6
df["is_active"] = True

source = ColumnDataSource(df)

# ====== СОЗДАЕМ ФИЛЬТРЫ ======
active_filter = BooleanFilter([True] * len(df))
inactive_filter = BooleanFilter([False] * len(df))

active_view = CDSView(filter=active_filter)
inactive_view = CDSView(filter=inactive_filter)


# ====== ФИГУРА КАРТЫ ======
output_file("phase_map.html")
p = figure(
    width=700, height=700,
    match_aspect=True,
    tools="pan,wheel_zoom,reset",
    title="Interactive GPT Phase Map",
    output_backend="webgl"
)
p.grid.visible = False


# ====== ФИГУРА ДЛЯ ГРАФИКА ЭНТРОПИИ ======
p_entropy = figure(
    width=600, height=350,
    tools="pan,wheel_zoom,reset",
    title="Entropy trajectory (select axis)",
    x_axis_label="Angle (°)",
    y_axis_label="Entropy"
)

# Пустой источник для графика энтропии
entropy_source = ColumnDataSource(data={
    'angle': [],
    'entropy': [],
    'regime': [],
    'color': [],
    'text': []
})

entropy_line = p_entropy.line('angle', 'entropy', source=entropy_source, 
                               line_width=2, color='gray', alpha=0.5)
entropy_circles = p_entropy.scatter('angle', 'entropy', source=entropy_source,
                                    size=8, color='color', alpha=0.9)

p_entropy.add_tools(HoverTool(
    renderers=[entropy_circles],
    tooltips=[
        ("Angle", "@angle°"),
        ("Entropy", "@entropy{0.00}"),
        ("Regime", "@regime"),
        ("Text", "@text")
    ]
))


# ====== КОНЦЕНТРИЧЕСКИЕ ОКРУЖНОСТИ ======
for deg, r in zip(degree_levels, radii_vals):
    if deg > 0:
        circle = BokehCircle(x=0, y=0, radius=r, fill_alpha=0, line_color='gray', line_alpha=0.15)
        p.add_glyph(circle)
        p.text(x=[r+0.01], y=[0], text=[f"{deg}°"], text_font_size="8pt", text_alpha=0.5)


# ====== ЛИНИИ МЕЖДУ ТОЧКАМИ ВДОЛЬ ОСЕЙ (ОПТИМИЗИРОВАНО) ======
by_dir_pos = defaultdict(list)
by_dir_neg = defaultdict(list)

for _, row in df.iterrows():
    d = int(row.dir)
    ang = float(row.angle)
    regime = row.regime
    x, y = row.x, row.y
    
    if x is None or pd.isna(x):
        continue
    
    r = get_radius(abs(ang))
    
    if ang >= 0:
        by_dir_pos[d].append((r, x, y, regime))
    else:
        by_dir_neg[d].append((r, x, y, regime))

def color_for(regime):
    return COLOR.get(regime, "lightgray")

# Собираем все линии по цветам для batch rendering
lines_by_color = defaultdict(lambda: {'xs': [], 'ys': []})

# + сторона
for d, pts in by_dir_pos.items():
    pts = sorted(pts, key=lambda t: t[0])
    for (r1,x1,y1,reg1), (r2,x2,y2,reg2) in zip(pts[:-1], pts[1:]):
        col = color_for(reg2)
        lines_by_color[col]['xs'].append([x1, x2])
        lines_by_color[col]['ys'].append([y1, y2])

# - сторона
for d, pts in by_dir_neg.items():
    pts = sorted(pts, key=lambda t: t[0])
    for (r1,x1,y1,reg1), (r2,x2,y2,reg2) in zip(pts[:-1], pts[1:]):
        col = color_for(reg2)
        lines_by_color[col]['xs'].append([x1, x2])
        lines_by_color[col]['ys'].append([y1, y2])

# Рисуем все линии одного цвета разом
for col, lines in lines_by_color.items():
    p.multi_line(lines['xs'], lines['ys'], color=col, line_width=1.2, alpha=0.8)


# ====== ТОЧКА В ЦЕНТРЕ ======
p.scatter([0], [0], size=8, color='black')


# ====== ТОЧКИ - НЕактивные (фон) ======
p.scatter(
    x='x', y='y',
    source=source,
    view=inactive_view,
    size='size',
    fill_color='color',
    fill_alpha=0.12,
    line_color=None,
)

# ====== ТОЧКИ - АКТИВНЫЕ (кликабельные) ======
r_active = p.scatter(
    x='x', y='y',
    source=source,
    view=active_view,
    size='size',
    fill_color='color',
    fill_alpha=0.95,
    line_color=None,
)


# ====== HOVER ======
p.add_tools(HoverTool(
    renderers=[r_active],
    tooltips=[
        ("dir", "@dir"),
        ("angle", "@angle"),
        ("text", "@text")
    ]
))

p.add_tools(TapTool())


# ====== ЛЕГЕНДА ======
legend_items = []
labels = ["BASE", "R", "B", "G"]
colors = ["black", "red", "blue", "green"]

for label, color, regime_hash in zip(labels, colors, top):
    dummy = p.scatter([], [], size=8, color=color, alpha=0.9)
    legend_items.append(LegendItem(label=f"{label}: {regime_hash[:8]}", renderers=[dummy]))

legend = Legend(items=legend_items, location="top_right")
p.add_layout(legend)


# ====== ТЕКСТ ПРИ КЛИКЕ ======
text_div = Div(text="<b>Кликни по точке</b>", width=600)

# ====== ВЫБОР ОСИ ======
plane_select = Select(
    title="Плоскость (dir):",
    value="ALL",
    options=["ALL"] + sorted(df["dir"].astype(str).unique().tolist())
)

base_x = df["x"].values.copy()
base_y = df["y"].values.copy()

# Преобразуем logs_data в JSON для передачи в JavaScript
logs_json = {}
for dir_num, entries in logs_data.items():
    logs_json[str(dir_num)] = entries

highlight_cb = CustomJS(
    args=dict(
        source=source,
        select=plane_select,
        active_filter=active_filter,
        inactive_filter=inactive_filter
    ),
    code="""
        const data = source.data;
        const dirs = data['dir'];
        const chosen = select.value;
        const act = active_filter.booleans;
        const inact = inactive_filter.booleans;

        for (let i = 0; i < dirs.length; i++) {
            if (chosen === "ALL") {
                act[i] = true;
                inact[i] = false;
            }
            else if (String(dirs[i]) === chosen) {
                act[i] = true;
                inact[i] = false;
            }
            else {
                act[i] = false;
                inact[i] = true;
            }
        }

        active_filter.change.emit();
        inactive_filter.change.emit();
        source.change.emit();
    """
)

plane_select.js_on_change("value", highlight_cb)


# JavaScript callback для обновления графика энтропии
callback = CustomJS(
    args=dict(
        source=source, 
        div=text_div, 
        select=plane_select,
        entropy_source=entropy_source,
        logs_data=logs_json,
        color_map=COLOR
    ), 
    code="""
    let inds = source.selected.indices;

    if (inds.length == 0) {
        div.text = "<b>Кликни по точке</b>";
        return;
    }

    let i = inds[inds.length - 1];
    const chosen = select.value;
    const dirs = source.data['dir'];

    if (chosen !== "ALL" && String(dirs[i]) !== chosen) {
        source.selected.indices = [];
        div.text = "<b>Кликни по точке</b>";
        return;
    }

    source.selected.indices = [i];

    const plane = source.data['dir'][i];
    const angle = source.data['angle'][i];
    const text = source.data['text'][i];

    div.text = `
        <p><b>dir:</b> ${plane}</p>
        <p><b>angle:</b> ${angle}°</p>
        <pre style="max-height: 200px; overflow-y: auto;">${text}</pre>
    `;

    // Обновляем график энтропии
    const dir_data = logs_data[String(plane)];
    if (dir_data) {
        const angles = [];
        const entropies = [];
        const regimes = [];
        const colors = [];
        const texts = [];

        for (let j = 0; j < dir_data.length; j++) {
            angles.push(dir_data[j].angle);
            entropies.push(dir_data[j].entropy);
            regimes.push(dir_data[j].regime);
            
            const regime = dir_data[j].regime;
            const color = color_map[regime] || "lightgray";
            colors.push(color);
            
            texts.push(dir_data[j].text.substring(0, 100));
        }

        entropy_source.data = {
            'angle': angles,
            'entropy': entropies,
            'regime': regimes,
            'color': colors,
            'text': texts
        };
        entropy_source.change.emit();
    }
""")

source.selected.js_on_change("indices", callback)


# ====== СБОРКА UI ======
layout = bokeh_row(
    p,
    bokeh_column(plane_select, text_div, p_entropy)
)

show(layout)