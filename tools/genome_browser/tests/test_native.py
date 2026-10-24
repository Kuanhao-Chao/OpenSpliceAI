import csv
import json
import math
import tempfile
import shutil
import sqlite3
import unittest
from pathlib import Path
import pyBigWig
from tools.genome_browser.build import prepare_parallel, finalize, file_sha
from tools.genome_browser.native import build_tracks, build_native_reference
from tools.genome_browser.format import canonical_json
from tools.genome_browser.hosting import verify
from tools.genome_browser.complete import adopt_checkpoint,builder_identity


class NativeTests(unittest.TestCase):
    def test_duplicate_shard_boundary_zero_missing_and_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ref=root/'ref.fa'; ref.write_text('>chr1\nACGTACGT\n'); Path(str(ref)+'.fai').write_text('chr1\t8\t6\t8\t9\n')
            genes=root/'genes.tsv'; genes.write_text('G\tchr1\t-\t0\t8\t0,4,\t2,8,\n')
            manifests={}
            for model in ('r10','r13'):
                path=root/f'{model}.tsv'; manifests[model+'_manifest']=str(path)
                with path.open('w') as f:
                    w=csv.DictWriter(f,delimiter='\t',fieldnames=['chunk_id','state','output_path','output_file_sha256','output_records']); w.writeheader()
                    for chunk in (1,2):
                        v=root/f'{model}-{chunk}.vcf'; score='0.00001' if model=='r10' else '0'
                        data=f'chr1\t1\t.\tA\tC\t.\t.\tOpenSpliceAI=C|G|{score}|0|0|0|-1|0|0|0\n'
                        if chunk==2: data+='chr1\t2\t.\tC\tG\t.\t.\t.\n'
                        v.write_text(data); w.writerow({'chunk_id':chunk,'state':'valid','output_path':str(v),'output_file_sha256':file_sha(v),'output_records':1 if chunk==1 else 2})
            config={**manifests,'id':'test-native','label':'test','reference':str(ref),'annotation':str(genes),'shard_size':1,'scope':'demonstration','models':{},'reference_review_regions':[{'chrom':'chr1','start':0,'end':8}]}
            output=root/'data';config['snapshot_directory']=str(output)
            code=root/'code/tools/genome_browser';shutil.copytree(Path(__file__).parents[1],code,ignore=shutil.ignore_patterns('tests','__pycache__'))
            config['builder_sha256']=builder_identity(code)
            cfg=root/'config.json'; cfg.write_bytes(canonical_json(config))
            prepare_parallel(cfg,output,workers=2); finalize(cfg,output); result=build_tracks(cfg,output)
            counts=result['coverage']['published']; self.assertEqual(counts['sourceOccurrences'],3); self.assertEqual(counts['uniqueCanonicalInput'],2)
            self.assertEqual(counts['models']['r13']['uniqueCanonicalPredicted'],1)
            with pyBigWig.open(str(output/'r13-AG.bw')) as b:
                values=b.values('chr1',0,2); self.assertEqual(values[0],0); self.assertTrue(math.isnan(values[1]))
            with pyBigWig.open(str(output/'coverage-r13.bw')) as b: self.assertEqual(b.values('chr1',0,2),[1,0])
            with pyBigWig.open(str(output/'r10-AG.bw')) as b: self.assertAlmostEqual(b.values('chr1',0,1)[0],0.00001,places=10)
            build_native_reference(cfg,output,excerpts=True); self.assertTrue(verify(output)['passed'])
            self.assertIn('>chr1:1-8 @len=8',(output/'reference-excerpts.fa').read_text())
            build_native_reference(cfg,output)
            self.assertTrue((output/'native-reference-receipt.json').exists())
            with (output/'reference.fa.bgz').open('ab') as f: f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'reference resume verification'): build_native_reference(cfg,output)
            newcfg=root/'next.json';nextconfig={**config,'id':'next','snapshot_directory':str(root/'next-data')};newcfg.write_bytes(canonical_json(nextconfig))
            self.assertEqual(adopt_checkpoint(newcfg,cfg),2)
            with sqlite3.connect(root/'next-data/preparation.sqlite') as db:self.assertEqual(db.execute('select sum(records) from shards').fetchone()[0],3)
            with (code/'native.py').open('a') as f:f.write('\n# altered builder\n')
            with self.assertRaisesRegex(ValueError,'frozen builder'):adopt_checkpoint(newcfg,cfg)
            # A resume must reauthenticate every output before accepting it.
            build_tracks(cfg,output)
            with (output/'r10-AG.bw').open('ab') as f: f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'resume verification'): build_tracks(cfg,output)
