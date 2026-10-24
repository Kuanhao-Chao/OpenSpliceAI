import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from tools.genome_browser.jobs import parse_shared_quota, quota_guard


class QuotaTests(unittest.TestCase):
    def test_live_gpfs_separator_and_in_doubt_reservations(self):
        raw = '''Disk quotas for group ssalzbe1 (gid 1018):
Filesystem Fileset type blocks quota limit in_doubt grace | files quota limit in_doubt grace Remarks
data root GRP 7.765T 10T 10T 10.69G none | 4153870 4194304 4194304 2106 none rockfish.storage
data rosei GRP none rockfish.storage
'''
        result = parse_shared_quota(raw)
        self.assertEqual(result['freeFiles'], 38328)
        self.assertAlmostEqual(result['freeBytes'], (10 - 7.765) * 1024**4 - 10.69 * 1024**3)
        self.assertEqual(parse_shared_quota(raw.replace('|', '')), result)

    def test_ambiguous_missing_or_unlimited_limits_fail_closed(self):
        row = 'data root GRP 1G 10G 10G 1M none | 10 100 100 1 none'
        for raw in ('', row + '\n' + row, row.replace('10G', 'none'), row.replace('GRP', 'USR')):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_shared_quota(raw)

    def test_resume_reserves_only_files_still_to_be_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CURRENT_STATUS.md').write_text('remaining: **2**')
            output = root / 'data'; output.mkdir()
            for i in range(10): (output / str(i)).touch()
            config = {'scoring_campaign': str(root), 'snapshot_directory': str(output)}
            raw = 'data root GRP 1T 10T 10T 0 none | 10000 20011 20011 0 none'
            with patch('tools.genome_browser.jobs.subprocess.check_output', return_value=raw):
                result = quota_guard(config, 14)
                self.assertEqual(result['pendingPreparationFiles'], 4)
                with self.assertRaisesRegex(ValueError, 'insufficient quota'):
                    quota_guard(config, 16)


if __name__ == '__main__':
    unittest.main()

class AccountingTests(unittest.TestCase):
    def test_billing_uses_allocated_units_and_elapsed_not_requested_workers(self):
        from tools.genome_browser.status import parse_accounting
        result=parse_accounting('123|RUNNING|3600|8|32G|cpu=8,mem=32G,billing=8|ssalzbe1|shared|\n')
        self.assertEqual(result['billedUnitHours'],8)
        self.assertEqual(result['allocatedGpus'],0)
        self.assertEqual(result['allocatedCpus'],8)

    def test_status_rejects_other_snapshot_marker_and_does_not_create_database(self):
        import json,tempfile
        from pathlib import Path
        from tools.genome_browser.status import campaign_status
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';(root/'config.json').write_text(json.dumps({'id':'current','snapshot_directory':str(data),'chunks':[1],'shard_size':1}))
            (root/'review-ready.json').write_text(json.dumps({'verification':{'passed':True,'dataset':'other'}}))
            result=campaign_status(root)
            self.assertFalse(result['releaseReady']);self.assertEqual(result['scorePacks']['completed'],0);self.assertFalse(data.exists())
