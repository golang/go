// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan_test

import (
	"internal/runtime/gc/scan"
	"testing"
)

func TestFilterNil(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		testFilterNil(t, []uintptr{}, []uintptr{})
	})
	t.Run("one", func(t *testing.T) {
		testFilterNil(t, []uintptr{4}, []uintptr{4})
	})
	t.Run("elimOne", func(t *testing.T) {
		testFilterNil(t, []uintptr{0}, []uintptr{})
	})
	t.Run("oneElimBegin", func(t *testing.T) {
		testFilterNil(t, []uintptr{0, 4}, []uintptr{4})
	})
	t.Run("oneElimEnd", func(t *testing.T) {
		testFilterNil(t, []uintptr{4, 0}, []uintptr{4})
	})
	t.Run("oneElimMultiBegin", func(t *testing.T) {
		testFilterNil(t, []uintptr{0, 0, 0, 4}, []uintptr{4})
	})
	t.Run("oneElimMultiEnd", func(t *testing.T) {
		testFilterNil(t, []uintptr{4, 0, 0, 0}, []uintptr{4})
	})
	t.Run("oneElimMulti", func(t *testing.T) {
		testFilterNil(t, []uintptr{0, 0, 0, 4, 0}, []uintptr{4})
	})
	t.Run("two", func(t *testing.T) {
		testFilterNil(t, []uintptr{5, 12}, []uintptr{5, 12})
	})
	t.Run("twoElimBegin", func(t *testing.T) {
		testFilterNil(t, []uintptr{0, 5, 12}, []uintptr{5, 12})
	})
	t.Run("twoElimMid", func(t *testing.T) {
		testFilterNil(t, []uintptr{5, 0, 12}, []uintptr{5, 12})
	})
	t.Run("twoElimEnd", func(t *testing.T) {
		testFilterNil(t, []uintptr{5, 12, 0}, []uintptr{5, 12})
	})
	t.Run("twoElimMulti", func(t *testing.T) {
		testFilterNil(t, []uintptr{0, 5, 0, 12, 0}, []uintptr{5, 12})
	})
	t.Run("Multi", func(t *testing.T) {
		testFilterNil(t, []uintptr{1, 5, 5, 0, 0, 0, 12, 0, 121, 5, 0}, []uintptr{1, 5, 5, 12, 121, 5})
	})
}

func testFilterNil(t *testing.T, buf, want []uintptr) {
	var bufp *uintptr
	if len(buf) != 0 {
		bufp = &buf[0]
	}
	n := scan.FilterNil(bufp, int32(len(buf)))
	if n > int32(len(buf)) {
		t.Errorf("bogus new length returned: %d > %d", n, len(buf))
		return
	}
	buf = buf[:n]
	if len(buf) != len(want) {
		t.Errorf("lengths differ: got %d, want %d", len(buf), len(want))
	}

	wantMap := make(map[uintptr]int)
	gotMap := make(map[uintptr]int)
	for _, p := range want {
		wantMap[p]++
	}
	for _, p := range buf {
		gotMap[p]++
	}
	for p, nWant := range wantMap {
		if nGot, ok := gotMap[p]; !ok {
			t.Errorf("want %d, but missing from output", p)
		} else if nGot != nWant {
			t.Errorf("want %d copies of %d, but got %d", nWant, p, nGot)
		}
	}
	for p := range gotMap {
		if _, ok := wantMap[p]; !ok {
			t.Errorf("got %d, but didn't want it", p)
		}
	}
	t.Logf("got:  %v", buf)
	t.Logf("want: %v", want)
}
