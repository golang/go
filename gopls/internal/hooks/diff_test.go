// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hooks

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"
	"unicode/utf8"

	"golang.org/x/tools/internal/lsp/diff/difftest"
)

func TestDiff(t *testing.T) {
	difftest.DiffTest(t, ComputeEdits)
}

func TestRepl(t *testing.T) {
	t.Skip("just for checking repl by looking at it")
	repl := initrepl(800)
	t.Errorf("%q", string(repl))
	t.Errorf("%d", len(repl))
}

func TestDisaster(t *testing.T) {
	a := "This is a string,(\u0995) just for basic functionality"
	b := "Ths is another string, (\u0996) to see if disaster will store stuff correctly"
	fname := disaster(a, b)
	buf, err := ioutil.ReadFile(fname)
	if err != nil {
		t.Errorf("error %v reading %s", err, fname)
	}
	var x, y string
	n, err := fmt.Sscanf(string(buf), "%s\n%s\n", &x, &y)
	if n != 2 {
		t.Errorf("got %d, expected 2", n)
		t.Logf("read %q", string(buf))
	}
	if a == x || b == y {
		t.Error("failed to encrypt")
	}
	err = os.Remove(fname)
	if err != nil {
		t.Errorf("%v removing %s", err, fname)
	}
	alen, blen := utf8.RuneCount([]byte(a)), utf8.RuneCount([]byte(b))
	xlen, ylen := utf8.RuneCount([]byte(x)), utf8.RuneCount([]byte(y))
	if alen != xlen {
		t.Errorf("a; got %d, expected %d", xlen, alen)
	}
	if blen != ylen {
		t.Errorf("b: got %d expected %d", ylen, blen)
	}
}
