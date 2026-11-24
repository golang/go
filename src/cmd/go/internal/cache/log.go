// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

// With GOEXPERIMENT=trimbuildcache, GOCACHE/cache.log records each trim attempt. Sample lines:
//
//	<time> trim: size trigger: 9.000 GiB -> 6.000 GiB in 2.123s, removed 9 of 48 files, base=6.000 GiB next=9.000 GiB
//	<time> trim: 24h trigger: 1.200 GiB -> 0.800 GiB in 1.52s, removed 12 of 56 files, base=0.800 GiB next=5.000 GiB
//	<time> trim: low disk trigger: 4.100 GiB -> 1.300 GiB in 810ms, removed 30 of 90 files, base=1.000 GiB next=5.000 GiB free=0.400 GiB (consider providing more space)
//
// Once cache.log reaches logRollSize, it rotates through cache.1.log and cache.2.log as needed,
// with an old cache.2.log being discarded. 'go clean -cache' does not delete the logs.

const logRollSize int64 = 128 << 10 // 128 KiB per log file (3 files max)

// appendLogLocked appends a timestamped line to cache.log, rotating the logs if needed.
//
// The caller must hold the trim.txt lock. Logging is best effort: errors are ignored
// and repeated failure to roll the logs will cause an eventual logging halt. An intermittent
// failure to roll might cause the active log to grow up to ~2x expected max size.
func appendLogLocked(dir string, now time.Time, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	line := fmt.Sprintf("%s %s\n", now.Format("2006/01/02 15:04:05"), msg)
	if len(line) > 2048 {
		line = line[:2048] + " [truncated]\n"
	}

	// First check if we need to roll our log files.
	log0 := filepath.Join(dir, "cache.log")
	info, err := os.Stat(log0)
	if err == nil && info.Size() >= logRollSize {
		// Past max size for the active log, so roll.
		err := rollLogLocked(dir)
		if err != nil && info.Size() >= 2*logRollSize {
			// Rotation failed and cache.log is well past expected size.
			// Something is wrong, don't log anything.
			return
		}
	}

	// Ready to log.
	f, err := os.OpenFile(log0, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o666)
	if err != nil {
		return
	}
	defer f.Close()
	f.WriteString(line)
}

// rollLogLocked rotates cache.log -> cache.1.log -> cache.2.log.
// The caller must hold the trim.txt lock.
func rollLogLocked(dir string) error {
	// We have one, two, or three log files and the goal is to preserve
	// the two most recent with new names.
	//
	// A rare mid-rotation crash or rename failure such as from a virus scanner
	// or similar still preserves the file contents, though can leave
	// a gap in the numbering until the next rotation heals the gap.
	log0 := filepath.Join(dir, "cache.log")
	log1 := filepath.Join(dir, "cache.1.log")
	log2 := filepath.Join(dir, "cache.2.log")

	// First, 1 to 2 if 1 exists.
	err := os.Rename(log1, log2)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		// Not existing is fine, but some other error occurred.
		return err
	}
	// Second, 0 to 1.
	return os.Rename(log0, log1)
}

// formatBytes formats a byte count as GiB for cache.log, such as "1.500 GiB".
func formatBytes(b int64) string {
	return fmt.Sprintf("%.3f GiB", float64(b)/(1<<30))
}

// formatEstimateDelta returns a parenthetical note if est differs from actual
// by more than 10% and 1 GiB, or "" otherwise.
func formatEstimateDelta(est, actual int64) string {
	delta := est - actual
	if absDelta := max(delta, -delta); absDelta <= actual/10 || absDelta <= 1<<30 {
		return ""
	}
	var pct string
	if actual > 0 {
		pct = fmt.Sprintf(" delta=%+.1f%%", 100*float64(delta)/float64(actual))
	}
	return fmt.Sprintf(" (prior estimate=%s actual=%s%s)", formatBytes(est), formatBytes(actual), pct)
}
