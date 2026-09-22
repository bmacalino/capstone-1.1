"""Repeat the paired grouped comparison with the three user-confirmed labels.

Reuse round 5's feature, fitting, calibration, and threshold-selection functions.
Keep historical artifacts and reports intact. Neither candidate is deployed.
"""
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from research.compare_product_risk_round5 import (
    ROOT, ACTIVE, SEED, ADJUDICATED, build_rows, read_json, sha256,
    connected_groups, clean_risk_text, checked_splits, inner_select, fit_model,
    positive_probability, risk_metrics, product_metrics, HybridManuscriptRiskModel,
)
from research.build_confirmed_product_risk_labels import build_confirmed_rows, DECISIONS, DESTINATION

OUTPUT = ROOT / 'outputs/product_risk_round6'
PREVIOUS = ROOT / 'outputs/product_risk_round5'
DOMAINS = ('overall', 'lazada_all', 'lazada_legacy', 'lazada_hard', 'firecs')


def masks(cohorts):
    return {'overall': np.ones(len(cohorts), bool), 'lazada_all': cohorts != 'firecs',
            **{c: cohorts == c for c in DOMAINS[2:]}}


def frozen_splits(rows, original, previous_predictions):
    texts = np.array([r['text'] for r in rows], dtype=object)
    groups = connected_groups(rows)
    previous = {r['review_id']: r for r in previous_predictions}
    old_labels = np.array([r['risk_detected'] for r in original])
    if [r['review_id'] for r in rows] != [r['review_id'] for r in original]:
        raise ValueError('Original row order changed')
    cleaned = defaultdict(set)
    for row, group in zip(rows, groups):
        cleaned[clean_risk_text(row['text'])].add(group)
    if any(len(value) > 1 for value in cleaned.values()):
        raise ValueError('Cross-group duplicate after text cleaning')
    splits = list(checked_splits(texts, old_labels, groups, np.arange(len(rows)), folds=5, seed=SEED))
    for fold, (_, test) in enumerate(splits, 1):
        if any(previous[rows[i]['review_id']]['fold'] != fold for i in test):
            raise ValueError('Outer folds differ from round 5')
    return splits


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def evaluate_metrics(rows, predictions):
    labels = np.array([r['risk_detected'] for r in rows])
    cohorts = np.array([r['cohort'] for r in rows])
    comparison = {}
    for domain, mask in masks(cohorts).items():
        comparison[domain] = {}
        for name in ('lr', 'svm'):
            pred = np.array([r['models'][name]['pred'] for r in predictions])
            prob = np.array([r['models'][name]['prob'] for r in predictions])
            if (pred < 0).any() or not np.isfinite(prob).all():
                raise ValueError('Incomplete outer predictions')
            comparison[domain][name] = risk_metrics(labels[mask], pred[mask], prob[mask])
    return comparison


def verify():
    report = read_json(OUTPUT / 'training_report.json')
    predictions = read_json(OUTPUT / 'oof_predictions.json')
    rows, _, confirmation = build_confirmed_rows()
    assert len(predictions) == len(rows) == 2424
    assert sum(r['risk_detected'] for r in rows) == 1197
    assert len({r['review_id'] for r in predictions}) == len(rows)
    assert report['confirmation'] == confirmation
    groups = connected_groups(rows)
    product_folds = defaultdict(set)
    group_folds = defaultdict(set)
    old_predictions = read_json(PREVIOUS / 'oof_predictions.json')
    splits = frozen_splits(rows, build_rows(), old_predictions)
    expected_folds = {rows[i]['review_id']: fold for fold, (_, test) in enumerate(splits, 1) for i in test}
    for i, (row, saved) in enumerate(zip(rows, predictions)):
        assert row['review_id'] == saved['review_id']
        assert row['risk_detected'] == saved['actual']
        assert row['product_id'] == saved['product_id']
        assert row['cohort'] == saved['cohort']
        assert str(groups[i]) == saved['group']
        assert expected_folds[row['review_id']] == saved['fold']
        group_folds[saved['group']].add(saved['fold'])
        if row['product_id']:
            product_folds[row['product_id']].add(saved['fold'])
        for name, value in saved['models'].items():
            threshold = report['folds'][name][saved['fold'] - 1]['selected']['threshold']
            assert 0 <= value['prob'] <= 1
            assert value['pred'] == int(value['prob'] >= threshold)
    assert all(len(folds) == 1 for folds in group_folds.values())
    assert all(len(folds) == 1 for folds in product_folds.values())
    assert evaluate_metrics(rows, predictions) == report['comparison']
    for path, expected in report['protected_hashes'].items():
        assert sha256(ROOT / path) == expected, path
    for path, expected in report['code_hashes'].items():
        assert sha256(ROOT / path) == expected, path
    active = joblib.load(ACTIVE)
    for info in report['artifacts'].values():
        assert sha256(ROOT / info['path']) == info['sha256']
        artifact = joblib.load(ROOT / info['path'])
        assert artifact['metadata']['risk_training_rows'] == 2424
        assert artifact['metadata']['risk_distribution'] == {'no_risk': 1227, 'risk': 1197}
        assert artifact['metadata']['risk_training_lazada_products'] == 27
        assert artifact['metadata']['confirmation'] == confirmation
        assert not artifact['metadata']['qualified_for_integration']
        assert joblib.hash(active['sentiment_pipeline']) == joblib.hash(artifact['sentiment_pipeline'])
        assert artifact['metadata']['score_formula'] == active['metadata']['score_formula']
    verification = {'rows': len(rows), 'lazada_products': len(product_folds),
                    'confirmed_decisions_applied': True, 'same_outer_folds_as_round5': True,
                    'recomputed_metrics_match': True, 'threshold_replay_matches': True,
                    'groups_disjoint': True, 'source_artifact_and_code_hashes_match': True,
                    'sentiment_and_score_formula_preserved': True, 'deployment_changed': False}
    write_json('verification.json', verification)
    print(json.dumps(verification, indent=2), flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'training_report.json').exists():
        raise FileExistsError('Completed round 6 already exists; verify it instead of overwriting.')
    rows, prior_audit, confirmation = build_confirmed_rows()
    # JSON stores the legacy category tuples as arrays; compare the serialized
    # representation while preserving exact text, IDs, and binary labels.
    if json.loads(json.dumps(rows)) != read_json(DESTINATION / 'development_rows.json'):
        raise ValueError('Saved confirmed dataset differs from rebuilt labels')
    original = build_rows()
    prior_report = read_json(PREVIOUS / 'training_report.json')
    paths = [ROOT / path for path in prior_report['protected_hashes']]
    paths += [DECISIONS, DESTINATION / 'development_rows.json', DESTINATION / 'manifest.json']
    paths += [PREVIOUS / name for name in ('training_report.json', 'oof_predictions.json', 'verification.json')]
    paths += [ROOT / info['path'] for info in prior_report['artifacts'].values()]
    protected = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    code_paths = [Path(__file__), ROOT / 'research/compare_product_risk_round5.py',
                  ROOT / 'research/build_confirmed_product_risk_labels.py',
                  ROOT / 'research/improve_product_risk_round4.py',
                  ROOT / 'sentiment-analysis/defaketive_sentiment/product_risk_context.py']
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}
    texts = np.array([r['text'] for r in rows], dtype=object)
    labels = np.array([r['risk_detected'] for r in rows])
    cohorts = np.array([r['cohort'] for r in rows])
    groups = connected_groups(rows)
    splits = frozen_splits(rows, original, read_json(PREVIOUS / 'oof_predictions.json'))
    results = {name: {'pred': np.full(len(rows), -1), 'prob': np.full(len(rows), np.nan), 'folds': []}
               for name in ('lr', 'svm')}
    fold_ids = np.full(len(rows), -1)
    print(f'Confirmed labels: {len(rows)} rows, {sum(labels)} risk, {len(rows)-sum(labels)} no risk', flush=True)
    for fold, (train, test) in enumerate(splits, 1):
        print(f'Outer fold {fold}/5', flush=True)
        selected, entries = inner_select(texts, labels, groups, cohorts, train, SEED + 100 * fold)
        fold_ids[test] = fold
        for name, choice in selected.items():
            model = fit_model(choice['config'], texts, labels, groups, train, SEED + 2000 + fold)
            result = results[name]
            result['prob'][test] = positive_probability(model, texts[test])
            result['pred'][test] = (result['prob'][test] >= choice['threshold']).astype(int)
            result['folds'].append({'fold': fold, 'selected': choice, 'inner_candidates': entries,
                                    'train_count': len(train), 'test_count': len(test)})
        write_json('progress.json', {'status': 'running', 'completed_outer_folds': fold})
    predictions = [{'review_id': r['review_id'], 'cohort': r['cohort'], 'product_id': r['product_id'],
                    'group': str(groups[i]), 'fold': int(fold_ids[i]), 'actual': int(labels[i]),
                    'models': {n: {'pred': int(v['pred'][i]), 'prob': float(v['prob'][i])} for n,v in results.items()}}
                   for i,r in enumerate(rows)]
    comparison = evaluate_metrics(rows, predictions)
    write_json('oof_predictions.json', predictions)
    write_json('outer_comparison.json', comparison)
    print('Outer evaluation complete; fitting final candidates.', flush=True)
    all_indices = np.arange(len(rows))
    final, entries = inner_select(texts, labels, groups, cohorts, all_indices, SEED + 900)
    active = joblib.load(ACTIVE)
    artifacts = {}
    for name, choice in final.items():
        model = fit_model(choice['config'], texts, labels, groups, all_indices, SEED + 3000)
        path = ROOT / f'sentiment-analysis/models/product_risk_round6_{name}_candidate.joblib'
        metadata = {'model_version': f'product-risk-round6-{name}-candidate',
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'deployment_status': 'development_candidate_not_deployed', 'qualified_for_integration': False,
                    'sentiment_source_metadata': active['metadata'], 'sentiment_source_artifact': str(ACTIVE.relative_to(ROOT)),
                    'risk_policy': 'Product-only binary risk; no categories or severity in score',
                    'score_formula': active['metadata']['score_formula'], 'risk_training_rows': len(rows),
                    'risk_training_lazada_products': len({r['product_id'] for r in rows if r['product_id']}),
                    'risk_training_cohorts': dict(Counter(cohorts)),
                    'risk_distribution': dict(Counter('risk' if y else 'no_risk' for y in labels)),
                    'risk_detection_threshold': choice['threshold'], 'risk_model_selection': choice,
                    'confirmation': confirmation, 'protected_hashes': protected, 'code_hashes': code_hashes}
        artifact = {'metadata': metadata, 'sentiment_pipeline': active['sentiment_pipeline'],
                    'risk_detection_pipeline': model, 'risk_detection_threshold': choice['threshold']}
        joblib.dump(artifact, path)
        loaded = joblib.load(path)
        assert np.allclose(positive_probability(model, texts[:30]), positive_probability(loaded['risk_detection_pipeline'], texts[:30]))
        smoke = HybridManuscriptRiskModel(path, allow_unqualified=True).analyze('Hindi gumagana ang isang earbud.')
        assert isinstance(smoke['risk']['detected'], bool) and np.isfinite(smoke['risk']['confidence'])
        artifacts[name] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'selection': choice}
    lazada = np.where(cohorts != 'firecs')[0]
    sampled_product = {n: product_metrics([rows[i] for i in lazada], v['pred'][lazada]) for n,v in results.items()}
    gates = {name: {f'{domain}_{metric}': comparison[domain][name]['risk']['recall'] >= .8 if metric == 'recall_80' else comparison[domain][name]['macro_f1'] >= .75
                    for domain in ('lazada_all', 'firecs') for metric in ('recall_80', 'macro_f1_75')}
             for name in ('lr', 'svm')}
    report = {'comparison': comparison, 'confirmation': confirmation, 'previous_label_audit': prior_audit,
              'folds': {n:v['folds'] for n,v in results.items()}, 'final_candidates': entries,
              'artifacts': artifacts, 'sampled_product_risk_contribution': sampled_product,
              'development_gates': gates, 'protected_hashes': protected, 'code_hashes': code_hashes,
              'deployment_changed': False, 'protocol': prior_report['protocol'],
              'limitations': [s for s in prior_report['limitations'] if 'pending confirmation' not in s] + [
                  'All three queried labels are confirmed. Other legacy policy conflicts may remain.',
                  'Round-5 metrics used different labels; compare algorithms within this run, not as a pure before/after model effect.']}
    write_json('training_report.json', report)
    lines = ['# Confirmed-label model comparison', '', 'All three user decisions applied. App and historical artifacts unchanged.', '',
             '| Dataset | Model | Risk precision | Risk recall | Macro-F1 |', '|---|---|---:|---:|---:|']
    for domain in ('lazada_all', 'lazada_hard', 'firecs'):
        for name, metrics in comparison[domain].items():
            lines.append(f"| {domain} | {name} | {metrics['risk']['precision']:.1%} | {metrics['risk']['recall']:.1%} | {metrics['macro_f1']:.1%} |")
    lines += ['', report['protocol'], '', '## Limitations', ''] + [f'- {s}' for s in report['limitations']]
    lines += ['', 'Reproduce: `.venv\\Scripts\\python.exe -m research.compare_product_risk_round6`',
              'Verify: `.venv\\Scripts\\python.exe -m research.compare_product_risk_round6 --verify`', '']
    (OUTPUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    verify()
    write_json('progress.json', {'status': 'complete'})
    print(json.dumps({'comparison': comparison, 'gates': gates}, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        verify() if args.verify else main()
