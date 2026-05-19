// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"io"
	"runtime/metrics"
	"testing"
)

// reuseFramesNonDefaultCount returns how many times the
// http2reuseframes=0 opt-out has been taken, as reported through
// runtime/metrics.
func reuseFramesNonDefaultCount(t *testing.T) uint64 {
	t.Helper()
	s := []metrics.Sample{{Name: "/godebug/non-default-behavior/http2reuseframes:events"}}
	metrics.Read(s)
	if s[0].Value.Kind() != metrics.KindUint64 {
		t.Fatalf("metric %s has kind %v, want KindUint64", s[0].Name, s[0].Value.Kind())
	}
	return s[0].Value.Uint64()
}

// TestSetReuseFramesFromGODEBUG verifies the http2reuseframes escape
// hatch on a bare Framer: reuse is on by default and with
// http2reuseframes=1, off with http2reuseframes=0, and only the
// opt-out is counted as a non-default behavior.
func TestSetReuseFramesFromGODEBUG(t *testing.T) {
	for _, tc := range []struct {
		godebug   string
		wantReuse bool
	}{
		{"", true},
		{"http2reuseframes=1", true},
		{"http2reuseframes=0", false},
	} {
		t.Run("GODEBUG="+tc.godebug, func(t *testing.T) {
			t.Setenv("GODEBUG", tc.godebug)
			before := reuseFramesNonDefaultCount(t)
			fr := NewFramer(io.Discard, nil)
			fr.setReuseFramesFromGODEBUG()
			if got := fr.frameCache != nil; got != tc.wantReuse {
				t.Errorf("frame reuse = %v, want %v", got, tc.wantReuse)
			}
			wantCounted := uint64(0)
			if !tc.wantReuse {
				wantCounted = 1
			}
			if got := reuseFramesNonDefaultCount(t) - before; got != wantCounted {
				t.Errorf("non-default behavior count grew by %d, want %d", got, wantCounted)
			}
		})
	}
}
