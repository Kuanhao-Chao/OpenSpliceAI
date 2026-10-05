"""Actual inference gives the same owned positions with and without FASTA splits."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openspliceai.predict.predict import predict_cli


@pytest.mark.integration
@pytest.mark.parametrize('strand', ['+','-'])
@pytest.mark.parametrize('hdf_threshold,predict_all', [(0,False),(0,True),(10**9,False),(10**9,True)])
def test_split_and_unsplit_coordinates_probabilities_match(tmp_path,model_80nt,strand,hdf_threshold,predict_all):
    rng=np.random.default_rng(2)
    sequence=''.join(rng.choice(list('ACGT'),1001))
    source=tmp_path/'input.fa'
    source.write_text(f'>gene NC_001.2:101-1101({strand})\n{sequence}\n')
    checkpoint=tmp_path/'model.pt'
    torch.save(model_80nt.state_dict(),checkpoint)
    outputs=[]
    for label,split_threshold in (('whole',2000),('split',237)):
        args=SimpleNamespace(output_dir=str(tmp_path/label),flanking_size=80,model=str(checkpoint),
            input_sequence=str(source),annotation_file=None,threshold=0.,debug=False,predict_all=predict_all,
            hdf_threshold=hdf_threshold,flush_threshold=1,split_threshold=split_threshold,chunk_size=2)
        predict_cli(args)
        result={}
        for bed in (tmp_path/label).rglob('*.bed'):
            for line in bed.read_text().splitlines():
                fields=line.split('\t')
                key=(bed.name,fields[0],int(fields[1]),int(fields[2]),fields[5])
                assert key not in result, 'duplicate source position in owned BED output'
                result[key]=float(fields[4])
        outputs.append(result)
    assert outputs[0].keys()==outputs[1].keys()
    assert len(outputs[0])==2001  # 1001 donors
    # The first acceptor has no preceding base.
    np.testing.assert_allclose(list(outputs[0].values()),[outputs[1][key] for key in outputs[0]],rtol=0,atol=1e-6)
