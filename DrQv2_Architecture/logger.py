import csv
from pathlib import Path


class Logger:
    def __init__(self, result_dir, filename="metrics.csv", flush_every=1):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.result_dir / filename
        self.flush_every = max(1, int(flush_every))

        self._fieldnames = ["step"]
        self._rows_written = 0
        self._file = None
        self._writer = None

        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self._fieldnames = list(header)

    def _open(self):
        if self._file is None:
            self._file = self.path.open("a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
            if self.path.stat().st_size == 0:
                self._writer.writeheader()
                self._file.flush()

    def _rewrite_with_new_header(self, new_fields):
        self._open()
        self._file.close()
        self._file = None
        self._writer = None

        rows = []
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_fields, restval="")
            w.writeheader()
            for r in rows:
                w.writerow(r)

        tmp.replace(self.path)
        self._fieldnames = list(new_fields)

    def log(self, step, **metrics):
        row = {"step": int(step)}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            row[str(k)] = v

        new_cols = [k for k in row.keys() if k not in self._fieldnames]
        if new_cols:
            new_fields = self._fieldnames + new_cols
            self._rewrite_with_new_header(new_fields)

        self._open()
        self._writer.writerow(row)
        self._rows_written += 1
        if self._rows_written % self.flush_every == 0:
            self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()