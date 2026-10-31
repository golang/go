// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"testing"
	"time"
)

// restoreGlobalDateCache saves globalDateCache and restores it when t finishes.
// These tests mutate package-global state and must not call t.Parallel().
func restoreGlobalDateCache(t *testing.T) {
	t.Helper()
	orig := globalDateCache.Load()
	t.Cleanup(func() {
		globalDateCache.Store(orig)
	})
}

func TestCachedDate(t *testing.T) {
	restoreGlobalDateCache(t)

	// Warm the cache so consecutive calls compare against a populated entry.
	cachedDate()

	d1 := cachedDate()
	d2 := cachedDate()

	if _, err := time.Parse(TimeFormat, d1); err != nil {
		t.Fatalf("d1 = %q not a valid date: %v", d1, err)
	}
	if _, err := time.Parse(TimeFormat, d2); err != nil {
		t.Fatalf("d2 = %q not a valid date: %v", d2, err)
	}

	// Allow up to 3 seconds of scheduling delay to avoid flaky failures in CI.
	now := time.Now().UTC()
	matched := false
	for i := 0; i <= 3; i++ {
		want := now.Add(-time.Duration(i) * time.Second).Format(TimeFormat)
		if d2 == want {
			matched = true
			break
		}
	}
	if !matched {
		t.Fatalf("cachedDate() = %q; want a formatted time within 3 seconds before %q", d2, now.Format(TimeFormat))
	}
}

func TestCachedDateUpdatesAfterExpiry(t *testing.T) {
	restoreGlobalDateCache(t)

	staleSec := time.Now().Unix() - 10
	const stale = "stale"
	globalDateCache.Store(&dateCache{sec: staleSec, str: stale})

	got := cachedDate()

	if got == stale {
		t.Fatal("cachedDate did not refresh stale cache")
	}
	if _, err := time.Parse(TimeFormat, got); err != nil {
		t.Fatalf("cachedDate() = %q: %v", got, err)
	}

	c := globalDateCache.Load()
	if c == nil {
		t.Fatal("globalDateCache does not contain a dateCache")
	}
	if c.str == stale {
		t.Fatal("expected cached date to update after expiry")
	}
	if c.sec <= staleSec {
		t.Fatalf("expected refreshed cache sec (%d) to be newer than stale sec (%d)", c.sec, staleSec)
	}
}

func TestCachedDateInitializesWhenEmpty(t *testing.T) {
	restoreGlobalDateCache(t)

	// Use nil to represent the empty/initial state for atomic.Pointer
	globalDateCache.Store(nil)

	got := cachedDate()

	if got == "" {
		t.Fatal("cachedDate returned empty string on cold start")
	}
	if _, err := time.Parse(TimeFormat, got); err != nil {
		t.Fatalf("cachedDate() = %q: %v", got, err)
	}

	c := globalDateCache.Load()
	if c == nil {
		t.Fatal("globalDateCache does not contain a dateCache after cold start")
	}
	if c.str == "" {
		t.Fatal("expected cached date to be populated after cold start")
	}
	if c.sec == 0 {
		t.Fatalf("expected initialized cache sec to be non-zero, got %d", c.sec)
	}
}
