"""Independent cache, held-out filtering, shard and FASTA integrity oracles."""
from pathlib import Path
from types import SimpleNamespace

import h5py
from gffutils.exceptions import FeatureNotFoundError
import numpy as np
import pytest

from openspliceai.create_data import paralogs, utils, verify_h5_file
from openspliceai.predict import predict as prediction
from tests.fixtures.synthetic import MINI_GFF


def columns(sequence='A'*100):
    return [['g'], ['chr1'], ['+'], ['1'], [str(len(sequence))], [sequence], ['0'*len(sequence)]]


def test_paralogy_coverage_uses_query_span_with_reference_deletions(tmp_path, monkeypatch):
    # 60 query bases aligned in a 120-base block. Block/query incorrectly
    # reports 120% coverage; the independent query-span oracle is 60%.
    hit = SimpleNamespace(mlen=110, blen=120, q_st=20, q_en=80)
    monkeypatch.setattr(paralogs.mp, 'Aligner', lambda *a, **k: SimpleNamespace(map=lambda seq: [hit]))
    _, held_out = paralogs.remove_paralogous_sequences(columns(), columns(), .8, .8, tmp_path, 'test')
    assert held_out == columns()
    assert float((tmp_path/'removed_paralogs_test.txt').read_text().split()[2]) == .6


def test_paralogy_index_failure_is_explicit_and_cleans_reference(tmp_path, monkeypatch):
    references = []
    monkeypatch.setattr(paralogs.mp, 'Aligner', lambda path, **kw: references.append(Path(path)))
    with pytest.raises(ValueError, match='index'):
        paralogs.remove_paralogous_sequences(columns(), columns(), .8, .8, tmp_path, 'test')
    assert references and not references[0].exists()


@pytest.mark.parametrize('train,test', [([[] for _ in range(7)], columns()), (columns(), [[] for _ in range(7)])])
def test_empty_paralogy_split_is_valid(tmp_path, train, test):
    assert paralogs.remove_paralogous_sequences(train, test, 1., 1., tmp_path, 'test') == (train, test)


@pytest.mark.parametrize('train,identity,coverage', [(columns(), -1, .5), (columns(), .5, float('nan')), (columns(''), .5, .5), ([[1]], .5, .5)])
def test_paralogy_rejects_invalid_inputs(tmp_path, train, identity, coverage):
    with pytest.raises(ValueError):
        paralogs.remove_paralogous_sequences(train, columns(), identity, coverage, tmp_path, 'test')


def test_annotation_cache_is_keyed_by_source_content(tmp_path):
    gff = tmp_path/'annotation.gff'
    cache = tmp_path/'annotation.db'
    gff.write_text(MINI_GFF)
    first = utils.create_or_load_db(gff, cache)
    first.conn.close()
    unchanged_time = cache.stat().st_mtime_ns
    same = utils.create_or_load_db(gff, cache)
    same.conn.close()
    assert cache.stat().st_mtime_ns == unchanged_time
    gff.write_text(MINI_GFF.replace('geneM', 'new_gene'))
    updated = utils.create_or_load_db(gff, cache)
    try:
        assert updated['new_gene'].featuretype == 'gene'
        with pytest.raises(FeatureNotFoundError):
            updated['geneM']
    finally:
        updated.conn.close()


def test_annotation_rebuild_failure_preserves_previous_cache(tmp_path, monkeypatch):
    gff, cache = tmp_path/'source.gff', tmp_path/'cache.db'
    gff.write_text(MINI_GFF)
    database = utils.create_or_load_db(gff, cache)
    database.conn.close()
    before = cache.read_bytes()
    gff.write_text(MINI_GFF+'# changed\n')
    def fail(*args, **kwargs):
        raise ValueError('parser failed')
    monkeypatch.setattr(utils.gffutils, 'create_db', fail)
    with pytest.raises(ValueError, match='parser failed'):
        utils.create_or_load_db(gff, cache)
    assert cache.read_bytes() == before
    assert not list(tmp_path.glob('cache.db.*'))


def encoded(handle, index, invalid=False):
    x = np.zeros((1, 84, 4), dtype=np.int8)
    y = np.zeros((1, 1, 4, 3), dtype=np.int8)
    x[..., 0], y[..., 0] = 1, 1
    if invalid:
        x[0, 0, 1] = 1
    handle[f'X{index}'], handle[f'Y{index}'] = x, y


def test_verification_checks_late_shards_and_empty_validation(tmp_path, capsys):
    args = SimpleNamespace(output_dir=tmp_path, biotype='all', chr_split='train-test')
    for split in ('test', 'train', 'validation'):
        with h5py.File(tmp_path/f'dataset_{split}.h5', 'w') as handle:
            if split != 'validation':
                encoded(handle, 4)
                encoded(handle, 10, invalid=split == 'train')
            handle.create_group('metadata')
    with pytest.raises(ValueError, match='one-hot'):
        verify_h5_file.verify_h5(args)
    with h5py.File(tmp_path/'dataset_train.h5', 'a') as handle:
        handle['X10'][0, 0, 1] = 0
    verify_h5_file.verify_h5(args)
    assert 'validation: empty split' in capsys.readouterr().out
    assert (tmp_path/'verify_train.png').exists()


@pytest.mark.parametrize('biotype,split', [('bad', 'test'), ('all', 'bad')])
def test_verification_validates_options(tmp_path, biotype, split):
    with pytest.raises(ValueError):
        verify_h5_file.verify_h5(SimpleNamespace(output_dir=tmp_path, biotype=biotype, chr_split=split))


def test_unsplit_fasta_rejects_inconsistent_coordinates_and_closes(tmp_path, monkeypatch):
    fasta = tmp_path/'input.fa'
    fasta.write_text('>gene NC_001:10-20(+)\nACGT\n')
    resources = []
    original = prediction.Fasta
    class TrackedFasta(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            resources.append(self)
    monkeypatch.setattr(prediction, 'Fasta', TrackedFasta)
    with pytest.raises(ValueError, match='span'):
        prediction.get_sequences(str(fasta), str(tmp_path)+'/', 80, split_fasta_threshold=100)
    assert resources[0].faidx.file.closed
