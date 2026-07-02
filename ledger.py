"""The ledger: write predictions down before running anything.

Every claim in this project follows the same protocol:
  1. predict(key, value) BEFORE the code that measures it runs
  2. measure(key, value) from the actual run
  3. scoreboard() renders the verdicts

Predictions persist to receipts/<name>.json so a prediction made on a
laptop can be checked against a measurement made on a TPU.
"""

import json
import time
from pathlib import Path

RECEIPTS = Path(__file__).parent / "receipts"


class Ledger:
    def __init__(self, name: str):
        self.path = RECEIPTS / f"{name}.json"
        self.path.parent.mkdir(exist_ok=True)
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self):
        self.path.write_text(json.dumps(self.entries, indent=2))

    def predict(self, key: str, value, unit: str = "", note: str = ""):
        e = self.entries.setdefault(key, {})
        e.update(predicted=value, unit=unit, note=note,
                 predicted_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._save()

    def predict_range(self, key: str, lo, hi, unit: str = "", note: str = ""):
        """For predictions with a known unknown: the measurement must land
        inside [lo, hi]. Prefer predict() once the unknown is measured."""
        e = self.entries.setdefault(key, {})
        e.update(predicted_lo=lo, predicted_hi=hi, unit=unit, note=note,
                 predicted_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._save()

    def measure(self, key: str, value):
        e = self.entries.setdefault(key, {})
        e.update(measured=value, measured_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._save()

    def scoreboard(self, tolerance: float = 0.30) -> bool:
        """Print predicted vs measured. Numeric entries pass within
        `tolerance` relative error; string entries must match exactly."""
        width = max((len(k) for k in self.entries), default=10)
        print(f"\n{'=' * (width + 58)}")
        print(f"{'LEDGER':<{width}}  {'predicted':>18}  {'measured':>18}  verdict")
        print(f"{'-' * (width + 58)}")
        all_ok = True
        for key, e in self.entries.items():
            pred, meas = e.get("predicted"), e.get("measured")
            unit = e.get("unit", "")
            if "predicted_lo" in e:
                pred = f"[{e['predicted_lo']}, {e['predicted_hi']}]"
                if meas is None:
                    line, ok = "(pending)", True
                else:
                    ok = e["predicted_lo"] <= meas <= e["predicted_hi"]
                    line = "IN BAND PASS" if ok else "OUT OF BAND FAIL"
            elif meas is None:
                line, ok = "(pending)", True
            elif isinstance(pred, str):
                ok = pred == meas
                line = "MATCH" if ok else "MISS"
            else:
                err = (meas - pred) / pred if pred else float("inf")
                ok = abs(err) <= tolerance
                line = f"{err:+.0%} {'PASS' if ok else 'FAIL'}"
            all_ok &= ok
            print(f"{key:<{width}}  {str(pred):>15} {unit:<2}  {str(meas):>15} {unit:<2}  {line}")
        print(f"{'=' * (width + 58)}\n")
        return all_ok
