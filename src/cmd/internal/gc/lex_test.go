// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "testing"

func TestInternConcat(t *testing.T) {
	fromKind := "T"
	toKind := "E"
	var s string
	n := testing.AllocsPerRun(100, func() {
		s = internConcat("conv", fromKind, "2", toKind)
	})
	if s != "convT2E" {
		t.Fatalf("internConcat(\"conv\", \"T\", \"2\", \"E\")=%q want %q", s, "convT2E")
	}
	if n > 0 {
		t.Errorf("internConcat allocs per run=%f", n)
	}
}
