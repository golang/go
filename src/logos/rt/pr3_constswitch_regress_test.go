// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"reflect"
	"testing"
)

func TestPR3ReinterpretMismatchedPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("Reinterpret mismatched types did not panic")
		}
	}()
	var b byte = 0xAB
	p := &b
	_ = Reinterpret[*[16]byte](p)
}

func TestPR3ReinterpretIdenticalControl(t *testing.T) {
	s := []int64{1, 2, 3}
	got := Reinterpret[[]int64](&s)
	if !reflect.DeepEqual(got, s) {
		t.Fatalf("identical Reinterpret = %v, want %v", got, s)
	}
}
