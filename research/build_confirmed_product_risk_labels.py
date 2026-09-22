"""Apply user-confirmed decisions without rewriting historical experiments."""
import json
from pathlib import Path

from research.compare_product_risk_round5 import ROOT, OUTPUT, build_rows, reconcile, read_json, sha256

DECISIONS = OUTPUT / 'confirmed_decisions.json'
DESTINATION = ROOT / 'outputs/product_risk_confirmed_labels'
EXPECTED_IDS = {'lazada-r1-0076', 'lazada-r1-0190', 'lazada-r1-0904'}


def apply_confirmed(rows, decisions):
    updated = [dict(row) for row in rows]
    indexed = {r['review_id']: r for r in updated}
    ids = [d['review_id'] for d in decisions]
    if len(ids) != 3 or set(ids) != EXPECTED_IDS or len(indexed) != len(rows):
        raise ValueError('Duplicate or unexpected review IDs')
    for decision in decisions:
        row = indexed[decision['review_id']]
        if row['review_text'] != decision['review_text']:
            raise ValueError('Confirmed review text does not match source')
        if type(decision['risk_detected']) is not int or decision['risk_detected'] not in (0, 1):
            raise ValueError('Risk label must be binary')
        if row['cohort'] != 'lazada_legacy':
            raise ValueError('Decision targets an unexpected cohort')
        row['risk_detected'] = decision['risk_detected']
        row['risk_annotation_source'] = 'user_confirmed_2026-09-20'
        row['risk_annotation_reason'] = decision['reason']
    return updated


def build_confirmed_rows():
    rows, audit = reconcile(build_rows())
    confirmation = read_json(DECISIONS)
    return apply_confirmed(rows, confirmation['decisions']), audit, confirmation


def main():
    historical = [OUTPUT / name for name in ('training_report.json', 'oof_predictions.json', 'label_audit.json')]
    historical_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in historical}
    rows, prior_audit, confirmation = build_confirmed_rows()
    original = build_rows()
    changed = sum(a['risk_detected'] != b['risk_detected'] for a,b in zip(original, rows))
    assert len(rows) == 2424 and changed == 31
    manifest = {
        'status': 'labels_confirmed_models_not_retrained',
        'rows': len(rows), 'risk_yes': sum(r['risk_detected'] for r in rows),
        'risk_no': sum(not r['risk_detected'] for r in rows),
        'changes_from_original_sources': changed,
        'previous_adjudications_restored': prior_audit['restored_count'],
        'user_confirmed_decisions': confirmation,
        'pending_ids_from_round5': [],
        'confirmation_sha256': sha256(DECISIONS),
        'historical_round5_hashes': historical_hashes,
        'round5_metrics_status': 'Historical results use previous labels; rerun required for current-label metrics.',
        'full_legacy_policy_audit_complete': False,
        'deployment_changed': False,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, value in [('development_rows.json', rows), ('manifest.json', manifest)]:
        (DESTINATION / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert all(sha256(ROOT / p) == h for p,h in historical_hashes.items())
    print(json.dumps({k: manifest[k] for k in ('status', 'rows', 'risk_yes', 'risk_no', 'changes_from_original_sources', 'deployment_changed')}, indent=2))


if __name__ == '__main__':
    main()
