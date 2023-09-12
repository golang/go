// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hooks

import (
	"os"
	"testing"

	"golang.org/x/tools/internal/diff/difftest"
)

func TestDiff(t *testing.T) {
	difftest.DiffTest(t, ComputeEdits)
}

func TestDisaster(t *testing.T) {
	a := "This is a string,(\u0995) just for basic\nfunctionality"
	b := "This is another string, (\u0996) to see if disaster will store stuff correctly"
	fname := disaster(a, b)
	buf, err := os.ReadFile(fname)
	if err != nil {
		t.Fatal(err)
	}
	if string(buf) != a+"\x00"+b {
		t.Error("failed to record original strings")
	}
	if err := os.Remove(fname); err != nil {
		t.Error(err)
	}
}
