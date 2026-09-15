import sys
import time

import httpx


class RateLimitTracker:
    """Tracks the rate of requests to ensure they do not exceed a specified limit."""

    def __init__(self, token: str | None = None):
        self.token = token
        self.limit = 5000 if token else 60
        self.remaining = self.limit
        self.reset_epoch = time.time() + 3600

        self._print_startup_banner()

    def wait_if_needed(self):
        """If remaining quota is exhausted, pauses execution with a live countdown timer."""
        if self.remaining > 0:
            return

        now = time.time()
        wait_seconds = max(1, int(self.reset_epoch - now))

        print(
            f"\n[RATE LIMIT REACHED] Waiting for reset ({self.limit} req/hr quota)..."
        )

        for seconds_left in range(wait_seconds, 0, -1):
            mins, secs = divmod(seconds_left, 60)
            sys.stdout.write(
                f"\r[PAUSED] Resuming in: {mins:02d}:{secs:02d} | Total Limit: {self.remaining}/{self.limit}"
            )
            sys.stdout.flush()
            time.sleep(1)

        print("\n[RESUMING] Rate limit reset window reached. Continuing ingestion...\n")
        self.remaining = self.limit

    def _print_startup_banner(self):
        if self.token:
            print(
                f"[INFO] GITHUB_TOKEN detected. Starting quota: {self.remaining}/{self.limit} req/hr."
            )
        else:
            print(
                f"[WARNING] No GITHUB_TOKEN found! Operating under unauthenticated limit: {self.remaining}/{self.limit} req/hr."
            )
            print("[HINT] Export GITHUB_TOKEN to increase limit to 5000 req/hr.")

    def update(self, headers: httpx.Headers):
        """Update the rate limit tracker based on response headers."""
        if "x-ratelimit-limit" in headers:
            self.limit = int(headers["x-ratelimit-limit"])
        if "x-ratelimit-remaining" in headers:
            self.remaining = int(headers["x-ratelimit-remaining"])
        if "x-ratelimit-reset" in headers:
            self.reset_epoch = int(headers["x-ratelimit-reset"])

    def display_status(self):
        now = time.time()
        time_left = max(0, int(self.reset_epoch - now))
        mins, secs = divmod(time_left, 60)

        print(
            f" -> Quota: [{self.remaining}/{self.limit}] | Reset in: {mins:02d}:{secs:02d}"
        )
