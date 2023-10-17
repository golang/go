// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embedtest_test

import (
	"embed"
	"os"
	"testing"
)

var (
	global2      = global
	concurrency2 = concurrency
	glass2       = glass
	sbig2        = sbig
	bbig2        = bbig
)

//go:embed testdata/*.txt
var global embed.FS

//go:embed c*txt
var concurrency string

//go:embed testdata/g*.txt
var glass []byte

//go:embed testdata/ascii.txt
var sbig string

//go:embed testdata/ascii.txt
var bbig []byte

func testFiles(t *testing.T, f embed.FS, name, data string) {
	t.Helper()
	d, err := f.ReadFile(name)
	if err != nil {
		t.Error(err)
		return
	}
	if string(d) != data {
		t.Errorf("read %v = %q, want %q", name, d, data)
	}
}

func testString(t *testing.T, s, name, data string) {
	t.Helper()
	if s != data {
		t.Errorf("%v = %q, want %q", name, s, data)
	}
}

func TestXGlobal(t *testing.T) {
	testFiles(t, global, "testdata/hello.txt", "hello, world\n")
	testString(t, concurrency, "concurrency", "Concurrency is not parallelism.\n")
	testString(t, string(glass), "glass", "I can eat glass and it doesn't hurt me.\n")
	testString(t, concurrency2, "concurrency2", "Concurrency is not parallelism.\n")
	testString(t, string(glass2), "glass2", "I can eat glass and it doesn't hurt me.\n")

	big, err := os.ReadFile("testdata/ascii.txt")
	if err != nil {
		t.Fatal(err)
	}
	testString(t, sbig, "sbig", string(big))
	testString(t, sbig2, "sbig2", string(big))
	testString(t, string(bbig), "bbig", string(big))
	testString(t, string(bbig2), "bbig", string(big))

	if t.Failed() {
		return
	}

	// Could check &glass[0] == &glass2[0] but also want to make sure write does not fault
	// (data must not be in read-only memory).
	old := glass[0]
	glass[0]++
	if glass2[0] != glass[0] {
		t.Fatalf("glass and glass2 do not share storage")
	}
	glass[0] = old

	// Could check &bbig[0] == &bbig2[0] but also want to make sure write does not fault
	// (data must not be in read-only memory).
	old = bbig[0]
	bbig[0]++
	if bbig2[0] != bbig[0] {
		t.Fatalf("bbig and bbig2 do not share storage")
	}
	bbig[0] = old
}
