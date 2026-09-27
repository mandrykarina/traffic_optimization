"""Распределение трафика: поиск вершин и графическое решение задачи ЛП."""

import argparse
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys


@dataclass(frozen=True)
class Constraint:
    """Ограничение a*x + b*y <= limit."""

    a: int
    b: int
    limit: int
    name: str


# Все ограничения, включая неотрицательность, записаны со знаком <=.
CONSTRAINTS = (
    Constraint(1, 1, 12, "Физические линки"),
    Constraint(2, 1, 18, "Маршрутизатор"),
    Constraint(-1, 0, 0, "Неотрицательность x"),
    Constraint(0, -1, 0, "Неотрицательность y"),
)
EFFECT = (3, 2)
Point = tuple[Fraction, Fraction]
PROJECT_DIR = Path(__file__).resolve().parent


def intersection(first: Constraint, second: Constraint) -> Point | None:
    """Пересечение границ двух ограничений по формулам Крамера."""
    determinant = first.a * second.b - second.a * first.b
    if determinant == 0:
        return None  # Параллельные или совпадающие прямые.
    x = Fraction(first.limit * second.b - second.limit * first.b, determinant)
    y = Fraction(first.a * second.limit - second.a * first.limit, determinant)
    return x, y


def is_feasible(point: Point) -> bool:
    x, y = point
    return all(c.a * x + c.b * y <= c.limit for c in CONSTRAINTS)


def objective(point: Point) -> Fraction:
    x, y = point
    return EFFECT[0] * x + EFFECT[1] * y


def find_vertices() -> list[Point]:
    """Перебрать пары границ, отсеять недопустимые точки и повторы."""
    vertices = set()
    for first, second in combinations(CONSTRAINTS, 2):
        point = intersection(first, second)
        if point is not None and is_feasible(point):
            vertices.add(point)
    return sorted(vertices)


def solve() -> tuple[list[Point], Point]:
    """Решить данную ограниченную задачу с непустой допустимой областью."""
    vertices = find_vertices()
    if not vertices:
        raise ValueError("Не найдены допустимые вершины. Проверьте модель.")
    return vertices, max(vertices, key=objective)


def make_summary(vertices: list[Point], best: Point) -> str:
    lines = [
        "РАСПРЕДЕЛЕНИЕ ТРАФИКА ПО ДВУМ КАНАЛАМ",
        "",
        "Максимизировать F = 3*x + 2*y",
        "Ограничения: x + y <= 12; 2*x + y <= 18; x >= 0; y >= 0",
        "",
        "Допустимые вершины:",
        f"{'Канал А (x)':>14} {'Канал Б (y)':>14} {'Эффект F':>12}",
    ]
    for point in vertices:
        lines.append(f"{str(point[0]):>14} {str(point[1]):>14} {str(objective(point)):>12}")
    x, y = best
    lines.extend([
        "",
        "ОПТИМАЛЬНОЕ РЕШЕНИЕ",
        f"Трафик по каналу А: {x}",
        f"Трафик по каналу Б: {y}",
        f"Максимальный полезный эффект: {objective(best)}",
        "",
        "Проверка ресурсов:",
    ])
    for c in CONSTRAINTS[:2]:
        used = c.a * x + c.b * y
        lines.append(f"{c.name}: использовано {used} из {c.limit}; остаток {c.limit - used}")
    lines.extend([
        "",
        "Доказательство: (x + y) + (2*x + y) <= 12 + 18, то есть F <= 30.",
        f"В найденной точке F = {objective(best)}. Верхняя граница достигнута.",
    ])
    return "\n".join(lines)


def save_plot(vertices: list[Point], best: Point, destination: Path) -> None:
    # Agg сохраняет рисунок без окон: работает и в PyCharm, и в терминале.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import atan2

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    try:
        center_x = sum(float(p[0]) for p in vertices) / len(vertices)
        center_y = sum(float(p[1]) for p in vertices) / len(vertices)
        ordered = sorted(vertices, key=lambda p: atan2(float(p[1]) - center_y,
                                                      float(p[0]) - center_x))
        ax.fill([float(p[0]) for p in ordered], [float(p[1]) for p in ordered],
                color="#cce7dc", alpha=0.85, label="Допустимая область", zorder=1)
        ax.plot([0, 12], [12, 0], color="#2563a6", linewidth=2,
                label="Линки: x + y = 12")
        ax.plot([0, 9], [18, 0], color="#c07822", linewidth=2,
                label="Маршрутизатор: 2x + y = 18")

        value = float(objective(best))
        ax.plot([0, value / EFFECT[0]], [value / EFFECT[1], 0],
                color="#bb3344", linestyle="--", linewidth=2,
                label=f"Целевая прямая: 3x + 2y = {value:g}")
        # Градиент (3, 2) указывает направление роста целевой функции.
        ax.annotate("", xy=(10.5, 11), xytext=(7.5, 9),
                    arrowprops={"arrowstyle": "->", "color": "#bb3344", "lw": 2})
        ax.text(7.4, 8.5, "Рост эффекта F", color="#bb3344")
        offsets = {(0, 0): (12, 12), (0, 12): (12, -22), (9, 0): (12, 12)}
        for p in vertices:
            x, y = map(float, p)
            if p == best:
                continue
            ax.scatter(x, y, color="#28343f", s=45, zorder=4)
            ax.annotate(f"({x:g}; {y:g}), F = {objective(p)}", (x, y),
                        xytext=offsets.get((x, y), (10, 10)), textcoords="offset points",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})
        bx, by = map(float, best)
        ax.scatter(bx, by, marker="*", s=280, color="#bb3344", zorder=5)
        ax.annotate(f"Оптимум ({bx:g}; {by:g})\nF = {value:g}", (bx, by),
                    xytext=(35, 25), textcoords="offset points", fontweight="bold",
                    arrowprops={"arrowstyle": "->", "color": "#bb3344"},
                    bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#bb3344"})
        ax.set(title="Распределение трафика по двум каналам",
               xlabel="x — трафик по каналу А, усл. ед.",
               ylabel="y — трафик по каналу Б, усл. ед.",
               xlim=(-0.7, 13), ylim=(-0.8, 19))
        ax.set_xticks(range(0, 13))
        ax.set_yticks(range(0, 19, 2))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=9)
        fig.savefig(destination, dpi=180)
    finally:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true",
                        help="выполнить только расчёт, без matplotlib и графика")
    args = parser.parse_args()
    vertices, best = solve()
    summary = make_summary(vertices, best)
    print(summary)
    output_dir = PROJECT_DIR / "results"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "solution.txt").write_text(summary + "\n", encoding="utf-8")
    except OSError as error:
        print(f"Не удалось сохранить результаты: {error}", file=sys.stderr)
        return 1
    if not args.no_plot:
        try:
            save_plot(vertices, best, output_dir / "feasible_region.png")
        except ImportError:
            print("Для графика установите зависимости: python -m pip install -r requirements.txt\n"
                  "Расчёт сохранён. Без графика: python main.py --no-plot", file=sys.stderr)
            return 1
        except OSError as error:
            print(f"Не удалось сохранить график: {error}", file=sys.stderr)
            return 1
        print(f"\nГрафик: {output_dir / 'feasible_region.png'}")
    print(f"Результаты: {output_dir / 'solution.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
