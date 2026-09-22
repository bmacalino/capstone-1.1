"""Finalize candidate provenance without retraining or modifying the live app.

Run after compare_product_risk_round5 and before verify_product_risk_round5.
The inherited sentiment metadata describes an older training run, not the new
risk head. Keep it explicitly separate instead of implying those counts apply
to the new risk classifier.
"""
from collections import Counter
import json
from pathlib import Path

import joblib

from research.compare_product_risk_round5 import ROOT, OUTPUT, ACTIVE, read_json, sha256, build_rows, reconcile


def main():
    report = read_json(OUTPUT / 'training_report.json')
    rows, audit = reconcile(build_rows())
    for path, expected in report['protected_hashes'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Protected source changed: {path}')
    active = joblib.load(ACTIVE)
    for name, info in report['artifacts'].items():
        path = ROOT / info['path']
        if sha256(path) != info['sha256']:
            raise ValueError('Candidate artifact changed unexpectedly')
        artifact = joblib.load(path)
        metadata = artifact['metadata']
        artifact['metadata'] = {
            'model_version': metadata['model_version'], 'created_at': metadata['created_at'],
            'sklearn_version': active['metadata']['sklearn_version'],
            'deployment_status': 'development_candidate_not_deployed',
            'qualified_for_integration': False,
            'qualification_note': 'Iterative grouped development comparison; label audit remains incomplete and fresh locked validation is required.',
            'sentiment_source_artifact': str(ACTIVE.relative_to(ROOT)),
            'sentiment_source_sha256': report['protected_hashes'][str(ACTIVE.relative_to(ROOT))],
            'sentiment_source_metadata': active['metadata'],
            'risk_policy': 'Product-only binary risk yes/no; no risk categories or severity in scoring',
            'score_formula': active['metadata']['score_formula'],
            'risk_training_rows': len(rows),
            'risk_training_cohorts': dict(Counter(r['cohort'] for r in rows)),
            'risk_distribution': dict(Counter('risk' if r['risk_detected'] else 'no_risk' for r in rows)),
            'risk_training_lazada_products': len({r['product_id'] for r in rows if r['product_id']}),
            'risk_detection_threshold': artifact['risk_detection_threshold'],
            'risk_model_selection': info['selection'],
            'label_overlay': audit, 'source_hashes': report['protected_hashes'],
        }
        assert joblib.hash(artifact['sentiment_pipeline']) == joblib.hash(active['sentiment_pipeline'])
        joblib.dump(artifact, path)
        info['sha256'] = sha256(path)
    report['metadata_finalization'] = {'script': str(Path(__file__).relative_to(ROOT)),
                                       'sha256': sha256(Path(__file__)), 'retrained': False}
    (OUTPUT / 'training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Candidate provenance finalized; classifier parameters and live app unchanged.')


if __name__ == '__main__':
    main()
