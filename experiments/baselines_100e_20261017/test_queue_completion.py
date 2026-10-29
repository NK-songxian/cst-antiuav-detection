"""Best.pt alone is not success: catch the previous queue's false completion."""
import json
import tempfile
from pathlib import Path

from run_baseline_queue import successful


with tempfile.TemporaryDirectory() as temp:
    path = Path(temp)
    best = path / 'best.pt'
    result = path / 'results.csv'
    best.write_bytes(b'checkpoint')
    assert not successful(-9, {}, result, best)
    assert not successful(0, {}, result, best)
    result.write_text('epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.3\n')
    assert not successful(-9, {'status': 'completed'}, result, best)
    assert not successful(0, {'status': 'failed'}, result, best)
    assert successful(0, {'status': 'completed'}, result, best)
    result.write_text('epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,nan,0.3\n')
    assert not successful(0, {'status': 'completed'}, result, best)
print(json.dumps({'queue_completion_checks': 'PASS', 'cases': 6}))
