"""Check copied evidence and declare a descriptive result card without inventing p-values."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE/'comparison.json').read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checks = []
for name, row in R['models'].items():
    path = HERE/'evaluations/test'/name
    audit = json.loads((path/'audit.json').read_text())
    assert sha(path/'audit.json') == row['provenance']['test_audit_sha256']
    assert sha(path/'frame_stats.npz') == row['provenance']['stats_sha256']
    assert sha(HERE/'evaluate_cst_completed.py') == audit['evaluation_source_sha256']
    assert audit['protocol']['split'] == 'test' and audit['groups']['ALL']['frames'] == 14446
    assert audit['groups']['ALL']['negative_frames'] == 2592
    assert audit['protocol']['actual_shapes'] == {'(3, 544, 672)': 14446}
    checks.append(dict(model=name, hash_verified=True, frames_verified=True, shape_verified=True))
assert len(checks) == 8
assert json.loads((HERE/'evaluation_status.json').read_text())['status'] == 'completed'
assert all(r['returncode'] == 0 for r in json.loads((HERE/'evaluation_status.json').read_text())['completed'])
template = Path('C:/Users/Ryan/.codex/skills/light-result-analysis/templates/result-card.example.json')
card = json.loads(template.read_text(encoding='utf-8'))
plan = json.loads((HERE/'evaluation_plan.json').read_text())
c1 = R['models']['C1_temporal_uniform']
base = R['models']['yolo11n']
now = datetime.now(timezone.utc).isoformat()
card.update(project='CST-AntiUAV-completed-20261017', as_of=now, decision='REVISION_REQUIRED')
card['plan_lock'] = dict(status='EXPLORATORY',locked_at=plan['started_at'],
    results_available_at=R['as_of'], locator=str(HERE/'evaluation_plan.json'),sha256=sha(HERE/'evaluation_plan.json'))
c = card['result_card']
c.update(claim_id='C1_test_AP50_descriptive', target_claim='Report the observed C1 versus YOLO11n test AP50 difference and false-alarm tradeoff.',
    target_type='SUPPLEMENTARY',estimand_or_hypothesis='Pooled held-out AP50 difference conditional on these fixed seed-0 weights.',
    metric='test mAP50 difference in percentage points', comparison_family='five_exploratory_fixed_checkpoint_contrasts',family_size=5,
    correction='NONE_NO_SIGNIFICANCE_TESTS',analysis_set=dict(name='all held-out CST test sequences',n=60,unit='video sequence',
        missingness=dict(rate=0,mechanism='NONE',handling='All 14446 held-out frames retained, including 2592 negatives.')))
c['provenance'] = dict(source_run_ids=['yolo11n','C1_temporal_uniform'],run_manifest_locator=str(HERE/'evaluation_status.json'),
    run_manifest_sha256=sha(HERE/'evaluation_status.json'),raw_result_locator=str(HERE/'comparison.json'),raw_result_sha256=sha(HERE/'comparison.json'),
    analysis_code_locator=str(HERE/'analyze_cst_completed.py'),analysis_code_sha256=sha(HERE/'analyze_cst_completed.py'),computed_at=now,owner_skill='RESULT-ANALYSIS')
c['practical_threshold'] = dict(metric='test AP50 difference pp',value=5,direction='HIGHER_IS_BETTER',
    justification_locator='User-requested 5-point target; post-training descriptive comparison, not a newly preregistered primary endpoint.')
c['assumption_checks'] = [dict(assumption='Train/val/test sequences are disjoint; past-only history within each split.',status='PASS',evidence_locator=str(HERE/'test_data_audit.json')),
    dict(assumption='Training seed variability is characterized.',status='UNKNOWN',evidence_locator='Only seed0 has been trained.')]
c['guardrail_analysis'] = dict(required=True,evidence_locator=str(HERE/'comparison.json'),evidence_sha256=sha(HERE/'comparison.json'),checks=[
    dict(id='val_BG_FP025',metric='val background frame FP rate at confidence0.25',status='FAIL',
         observed=c1['val_background_fp025']['background_frame_fp_rate'],threshold=base['val_background_fp025']['background_frame_fp_rate']+.02,
         claim_impact='Do not claim unqualified false-alarm suppression at fixed confidence; report the validation-calibrated threshold analysis separately.',evidence_locator=str(HERE/'comparison.json'))])
c['effect'] = dict(estimate=c1['test_delta_ap50_pp'],effect_size=c1['test_delta_ap50_pp'],
    ci95=R['bootstrap']['contrasts']['C1_temporal_uniform__minus__yolo11n']['ci95_pp'],p=None,q=None,direction='POSITIVE',practical_significance='DESCRIPTIVE_ONLY')
c['language'] = dict(conclusion='The fixed seed-0 weights show an observed test AP50 difference of 13.46 percentage points. Sequence bootstrap is conditional; no confirmatory p-values or SOTA claim.',
    strength='DESCRIPTIVE_ONLY',near_threshold_wording='No significance threshold decision is made; no p-values are fabricated.')
card['decision_ledger'] = [dict(decision_id='D1',timing='PRE',kind='MODEL_CHOICE',planned=True,classification='EXPLORATORY',
    affected_claim_ids=[c['claim_id']],rationale='Evaluate every native validation-AP95 best.pt; no test checkpoint selection.',evidence_locator=str(HERE/'evaluation_plan.json')),
    dict(decision_id='D2',timing='POST',kind='ANALYSIS',planned=False,classification='SENSITIVITY',affected_claim_ids=[c['claim_id']],
    rationale='Sequence bootstrap added as conditional descriptive sensitivity, not a confirmatory training-seed analysis.',evidence_locator=str(HERE/'comparison.json'))]
card['sensitivity'] = [dict(sensitivity_id='S1',classification='SENSITIVITY',description=R['bootstrap']['method'],evidence_locator=str(HERE/'comparison.json'))]
(HERE/'result_card.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
(HERE/'delivery_integrity.json').write_text(json.dumps(dict(status='PASS',checks=checks,as_of=now,
    scope='Copied metric artifacts, evaluator hashes, all 11 evaluation task exits and held-out frame inventory; not an originality or submission approval.'),indent=2),encoding='utf-8')
print('EVIDENCE_INTEGRITY_PASS',len(checks),'models; result card remains descriptive and requires revision before inferential publication claims.')
