from __future__ import annotations

import itertools
import json
from pathlib import Path

from umei.utils import DataKey

DATASET_ROOT = Path(__file__).parent
DATA_DIR = DATASET_ROOT / 'origin'

def load_cohort(task_id: int, merge: bool = False):
    cohort = {
        'training': {},
        'test': {}
    }
    # 1: MRI, 0: CT
    for modality, task in [(1, 2), (0, 1)]:
        if modality == 1 and task_id == 1:
            continue
        with open(DATA_DIR / f'task{task}_dataset.json') as f:
            task = json.load(f)
        for split in ['training', 'test']:
            for case in task[split]:
                if split == 'training':
                    img_path = Path(case['image'])
                    seg_path = Path(case['label'])
                else:
                    img_path = Path(case)
                    seg_path = None
                subject = img_path.name[:-7]
                cohort[split].update({
                    subject: {
                        'subject': subject,
                        'modality': modality,
                        DataKey.IMG: DATA_DIR / img_path,
                        **({} if seg_path is None else {'seg': DATA_DIR / seg_path if seg_path else None})
                    }
                })
    for split in ['training', 'test']:
        cohort[split] = list(cohort[split].values())
    if merge:
        return [
            {DataKey.IMG: x['img']}
            for x in itertools.chain(cohort['training'], cohort['test'])
        ]
    else:
        return cohort
