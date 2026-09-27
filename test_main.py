"""Проверки модели, граничных случаев пересечений и оптимального решения."""

from fractions import Fraction
import unittest

from main import Constraint, find_vertices, intersection, is_feasible, objective, solve


class TrafficOptimizationTests(unittest.TestCase):
    def test_all_vertices(self):
        self.assertEqual(set(find_vertices()), {(0, 0), (0, 12), (6, 6), (9, 0)})

    def test_optimal_solution_and_upper_bound(self):
        vertices, best = solve()
        self.assertEqual(best, (6, 6))
        self.assertEqual(objective(best), 30)
        for x, y in vertices:
            self.assertLessEqual((x + y) + (2 * x + y), 30)
        self.assertEqual(best[0] + best[1], 12)
        self.assertEqual(2 * best[0] + best[1], 18)

    def test_rejects_each_constraint_violation(self):
        # Нарушены соответственно: линки, маршрутизатор, x >= 0, y >= 0.
        for point in [(0, 13), (8, 4), (-1, 0), (0, -1)]:
            with self.subTest(point=point):
                self.assertFalse(is_feasible(point))

    def test_accepts_boundary_and_fractional_points(self):
        for point in [(0, 0), (9, 0), (6, 6), (0, 12),
                      (Fraction(1, 2), Fraction(3, 2))]:
            with self.subTest(point=point):
                self.assertTrue(is_feasible(point))

    def test_parallel_and_coincident_lines(self):
        first = Constraint(1, 1, 12, "")
        self.assertIsNone(intersection(first, Constraint(2, 2, 18, "")))
        self.assertIsNone(intersection(first, Constraint(2, 2, 24, "")))

    def test_exact_fractional_intersection(self):
        point = intersection(Constraint(2, 1, 4, ""), Constraint(1, 2, 4, ""))
        self.assertEqual(point, (Fraction(4, 3), Fraction(4, 3)))


if __name__ == "__main__":
    unittest.main()
