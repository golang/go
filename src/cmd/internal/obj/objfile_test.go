// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"bytes"
	"cmd/internal/goobj"
	"cmd/internal/sys"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"unsafe"
)

var dummyArch = LinkArch{Arch: sys.ArchAMD64}

func TestContentHash64(t *testing.T) {
	s1 := &LSym{P: []byte("A")}
	s2 := &LSym{P: []byte("A\x00\x00\x00")}
	s1.Set(AttrContentAddressable, true)
	s2.Set(AttrContentAddressable, true)
	h1 := contentHash64(s1)
	h2 := contentHash64(s2)
	if h1 != h2 {
		t.Errorf("contentHash64(s1)=%x, contentHash64(s2)=%x, expect equal", h1, h2)
	}

	ctxt := Linknew(&dummyArch) // little endian
	s3 := ctxt.Int64Sym(int64('A'))
	h3 := contentHash64(s3)
	if h1 != h3 {
		t.Errorf("contentHash64(s1)=%x, contentHash64(s3)=%x, expect equal", h1, h3)
	}
}

func TestContentHash(t *testing.T) {
	syms := []*LSym{
		&LSym{P: []byte("TestSymbol")},  // 0
		&LSym{P: []byte("TestSymbol")},  // 1
		&LSym{P: []byte("TestSymbol2")}, // 2
		&LSym{P: []byte("")},            // 3
		&LSym{P: []byte("")},            // 4
		&LSym{P: []byte("")},            // 5
		&LSym{P: []byte("")},            // 6
	}
	for _, s := range syms {
		s.Set(AttrContentAddressable, true)
		s.PkgIdx = goobj.PkgIdxHashed
	}
	// s3 references s0
	r := Addrel(syms[3])
	r.Sym = syms[0]
	// s4 references s0
	r = Addrel(syms[4])
	r.Sym = syms[0]
	// s5 references s1
	r = Addrel(syms[5])
	r.Sym = syms[1]
	// s6 references s2
	r = Addrel(syms[6])
	r.Sym = syms[2]

	// compute hashes
	h := make([]goobj.HashType, len(syms))
	w := &writer{}
	for i := range h {
		h[i] = w.contentHash(syms[i])
	}

	tests := []struct {
		a, b  int
		equal bool
	}{
		{0, 1, true},  // same contents, no relocs
		{0, 2, false}, // different contents
		{3, 4, true},  // same contents, same relocs
		{3, 5, true},  // recursively same contents
		{3, 6, false}, // same contents, different relocs
	}
	for _, test := range tests {
		if (h[test.a] == h[test.b]) != test.equal {
			eq := "equal"
			if !test.equal {
				eq = "not equal"
			}
			t.Errorf("h%d=%x, h%d=%x, expect %s", test.a, h[test.a], test.b, h[test.b], eq)
		}
	}
}

func TestSymbolTooLarge(t *testing.T) { // Issue 42054
	testenv.MustHaveGoBuild(t)
	if unsafe.Sizeof(uintptr(0)) < 8 {
		t.Skip("skip on 32-bit architectures")
	}

	tmpdir, err := ioutil.TempDir("", "TestSymbolTooLarge")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "p.go")
	err = ioutil.WriteFile(src, []byte("package p; var x [1<<32]byte"), 0666)
	if err != nil {
		t.Fatalf("failed to write source file: %v\n", err)
	}
	obj := filepath.Join(tmpdir, "p.o")
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "compile", "-o", obj, src)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("did not fail\noutput: %s", out)
	}
	const want = "symbol too large"
	if !bytes.Contains(out, []byte(want)) {
		t.Errorf("unexpected error message: want: %q, got: %s", want, out)
	}
}
