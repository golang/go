// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"iter"
	"testing"
)

type issue80714Kind int

//go:noinline
func (k issue80714Kind) each(yield func(int) bool) {
	_ = yield(int(k)) && yield(int(k)+1)
}

func (k issue80714Kind) seq() iter.Seq[int] {
	return k.each
}

func TestIssue80714(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	if n := testing.AllocsPerRun(10, func() {
		count := 0
		for v := range issue80714Kind(42).seq() {
			if v != 42 {
				t.Fatalf("yielded %d, want 42", v)
			}
			count++
			break
		}
		if count != 1 {
			t.Fatalf("yielded %d times, want 1", count)
		}
	}); n != 0 {
		t.Fatalf("unexpected allocations: %v", n)
	}
}
