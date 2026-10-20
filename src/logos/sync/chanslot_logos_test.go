// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"runtime"
	"testing"
	"weak"
)

type chanSlotPayload struct {
	a, b, c, d int64
	next       *chanSlotPayload
}

// TestLogosBufferedChanSlotCleared pins SPEC Sec 17 #143: Go 1.27 clears
// a received buffered-channel slot, so a received pointer becomes
// collectable once the receiver drops it. It runs in -short mode.
func TestLogosBufferedChanSlotCleared(t *testing.T) {
	ch := make(chan *chanSlotPayload, 1)
	v := &chanSlotPayload{a: 1, b: 2, c: 3, d: 4, next: &chanSlotPayload{a: 5}}
	w := weak.Make(v)
	ch <- v
	got := <-ch
	if got != v {
		t.Fatalf("received %p, want sent %p", got, v)
	}
	v = nil
	got = nil
	for range 10 {
		runtime.GC()
		if w.Value() == nil {
			return
		}
	}
	t.Fatalf("received pointer still reachable after receive, drop, and GC")
}
