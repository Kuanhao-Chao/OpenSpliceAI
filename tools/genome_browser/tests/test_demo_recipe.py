import copy
import unittest

from tools.genome_browser.demo_recipe import recipe_settings


class DemoRecipeTests(unittest.TestCase):
    def setUp(self):
        self.contigs = {'chr1': 40000, 'chrM': 16569}
        self.recipe = {'name': 'examples-v4', 'chunks': [1, 3],
                       'expected_source_occurrences': 6,
                       'default_view': {'chrom': 'chr1', 'start': 100, 'end': 120},
                       'regions': [{'chrom': 'chr1', 'start': 90, 'end': 180, 'label': 'Gene A'},
                                   {'chrom': 'chr1', 'start': 16000, 'end': 18000, 'label': 'Gene B'}]}

    def test_merged_reference_windows_and_separate_scored_intervals(self):
        original = copy.deepcopy(self.recipe)
        result = recipe_settings(self.recipe, self.contigs)
        self.assertEqual(result['reference_review_regions'], [
            {'chrom': 'chr1', 'start': 0, 'end': 32768},
            {'chrom': 'chrM', 'start': 0, 'end': 16569}])
        self.assertEqual(result['bookmarks'][1]['label'], 'Gene B')
        self.assertNotIn('label', result['regions'][1])
        self.assertEqual(self.recipe, original)

    def test_source_order_and_integer_ids_are_required(self):
        for chunks in ([], [3, 1], [1, 1], [True], [1.5], [0]):
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                recipe_settings({**self.recipe, 'chunks': chunks}, self.contigs)

    def test_invalid_bounds_contigs_labels_and_duplicate_regions(self):
        for changes in ({'start': -1}, {'end': 40001}, {'start': 180}, {'start': True},
                        {'chrom': 'chrUnknown'}, {'label': ''}):
            recipe = copy.deepcopy(self.recipe)
            recipe['regions'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                recipe_settings(recipe, self.contigs)
        recipe = copy.deepcopy(self.recipe)
        recipe['regions'].append(recipe['regions'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            recipe_settings(recipe, self.contigs)

    def test_default_scope_and_snapshot_name_cannot_escape_selection(self):
        for changes in ({'name': '../replace'}, {'expected_source_occurrences': True},
                        {'default_view': {'chrom': 'chr1', 'start': 0, 'end': 500}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                recipe_settings({**self.recipe, **changes}, self.contigs)
