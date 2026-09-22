"""Independently recompute saved metrics and verify round-5 provenance."""
import json
from collections import defaultdict

import numpy as np
import joblib

from research.compare_product_risk_round5 import (
    ROOT, OUTPUT, build_rows, reconcile, read_json, sha256, risk_metrics,
)


def main():
    report = read_json(OUTPUT / 'training_report.json')
    predictions = read_json(OUTPUT / 'oof_predictions.json')
    rows, audit = reconcile(build_rows())
    assert len(predictions) == len(rows) == 2424
    assert len({r['review_id'] for r in predictions}) == 2424
    assert audit == report['label_audit']
    labels = np.array([r['risk_detected'] for r in rows])
    cohorts = np.array([r['cohort'] for r in rows])
    seen_groups = defaultdict(set)
    seen_products = defaultdict(set)
    for row, saved in zip(rows, predictions):
        assert row['review_id'] == saved['review_id']
        assert row['risk_detected'] == saved['actual']
        assert row['product_id'] == saved['product_id']
        seen_groups[saved['group']].add(saved['fold'])
        if row['product_id']:
            seen_products[row['product_id']].add(saved['fold'])
        for name, result in saved['models'].items():
            threshold = report['folds'][name][saved['fold'] - 1]['selected']['threshold']
            assert 0 <= result['prob'] <= 1
            assert result['pred'] == int(result['prob'] >= threshold)
    assert all(len(folds) == 1 for folds in seen_groups.values())
    assert all(len(folds) == 1 for folds in seen_products.values())
    for domain, mask in [('overall', np.ones(len(rows), bool)), ('lazada_all', cohorts != 'firecs')] + [(c, cohorts == c) for c in ('lazada_legacy', 'lazada_hard', 'firecs')]:
        for name in ('lr', 'svm'):
            pred = np.array([r['models'][name]['pred'] for r in predictions])
            prob = np.array([r['models'][name]['prob'] for r in predictions])
            assert risk_metrics(labels[mask], pred[mask], prob[mask]) == report['comparison'][domain][name]
    for path, expected in report['protected_hashes'].items():
        assert sha256(ROOT / path) == expected, path
    for artifact in report['artifacts'].values():
        assert sha256(ROOT / artifact['path']) == artifact['sha256']
        metadata = joblib.load(ROOT / artifact['path'])['metadata']
        assert metadata['risk_training_rows'] == 2424
        assert metadata['risk_training_lazada_products'] == 27
        assert not metadata['qualified_for_integration']
        assert metadata['score_formula'] == '100 * (0.30 * negative_ratio + 0.70 * detected_risk_ratio)'
    finalized = report['metadata_finalization']
    assert sha256(ROOT / finalized['script']) == finalized['sha256']
    assert sha256(ROOT / 'research/compare_product_risk_round5.py') == report['code_sha256']
    verification = {'rows': len(rows), 'lazada_products': len(seen_products),
                    'restored_labels': audit['restored_count'], 'recomputed_metrics_match': True,
                    'outer_groups_and_products_disjoint': True, 'threshold_replay_matches': True,
                    'source_and_artifact_hashes_match': True, 'deployment_changed': False}
    (OUTPUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
