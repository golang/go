// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj

import (
	"bufio"
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/objabi"
	"fmt"
	"internal/buildcfg"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"testing"
)

func dummyWriter(buf *bytes.Buffer) *Writer {
	wr := &bio.Writer{Writer: bufio.NewWriter(buf)} // hacky: no file, so cannot seek
	return NewWriter(wr)
}

func TestReadWrite(t *testing.T) {
	// Test that we get the same data in a write-read roundtrip.

	// Write a symbol, a relocation, and an aux info.
	var buf bytes.Buffer
	w := dummyWriter(&buf)

	var s Sym
	s.SetABI(1)
	s.SetType(uint8(objabi.STEXT))
	s.SetFlag(0x12)
	s.SetSiz(12345)
	s.SetAlign(8)
	s.Write(w)

	var r Reloc
	r.SetOff(12)
	r.SetSiz(4)
	r.SetType(uint16(objabi.R_ADDR))
	r.SetAdd(54321)
	r.SetSym(SymRef{11, 22})
	r.Write(w)

	var a Aux
	a.SetType(AuxFuncInfo)
	a.SetSym(SymRef{33, 44})
	a.Write(w)

	w.wr.Flush()

	// Read them back and check.
	b := buf.Bytes()
	var s2 Sym
	s2.fromBytes(b)
	if s2.ABI() != 1 || s2.Type() != uint8(objabi.STEXT) || s2.Flag() != 0x12 || s2.Siz() != 12345 || s2.Align() != 8 {
		t.Errorf("read Sym2 mismatch: got %v %v %v %v %v", s2.ABI(), s2.Type(), s2.Flag(), s2.Siz(), s2.Align())
	}

	b = b[SymSize:]
	var r2 Reloc
	r2.fromBytes(b)
	if r2.Off() != 12 || r2.Siz() != 4 || r2.Type() != uint16(objabi.R_ADDR) || r2.Add() != 54321 || r2.Sym() != (SymRef{11, 22}) {
		t.Errorf("read Reloc2 mismatch: got %v %v %v %v %v", r2.Off(), r2.Siz(), r2.Type(), r2.Add(), r2.Sym())
	}

	b = b[RelocSize:]
	var a2 Aux
	a2.fromBytes(b)
	if a2.Type() != AuxFuncInfo || a2.Sym() != (SymRef{33, 44}) {
		t.Errorf("read Aux2 mismatch: got %v %v", a2.Type(), a2.Sym())
	}
}

var issue41621prolog = `
package main
var lines = []string{
`

var issue41621epilog = `
}
func getLines() []string {
	return lines
}
func main() {
	println(getLines())
}
`

func TestIssue41621LargeNumberOfRelocations(t *testing.T) {
	if testing.Short() || (buildcfg.GOARCH != "amd64") {
		t.Skipf("Skipping large number of relocations test in short mode or on %s", buildcfg.GOARCH)
	}
	testenv.MustHaveGoBuild(t)

	tmpdir, err := ioutil.TempDir("", "lotsofrelocs")
	if err != nil {
		t.Fatalf("can't create temp directory: %v\n", err)
	}
	defer os.RemoveAll(tmpdir)

	// Emit testcase.
	var w bytes.Buffer
	fmt.Fprintf(&w, issue41621prolog)
	for i := 0; i < 1048576+13; i++ {
		fmt.Fprintf(&w, "\t\"%d\",\n", i)
	}
	fmt.Fprintf(&w, issue41621epilog)
	err = ioutil.WriteFile(tmpdir+"/large.go", w.Bytes(), 0666)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	// Emit go.mod
	w.Reset()
	fmt.Fprintf(&w, "module issue41621\n\ngo 1.12\n")
	err = ioutil.WriteFile(tmpdir+"/go.mod", w.Bytes(), 0666)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}
	w.Reset()

	// Build.
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "large")
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Build failed: %v, output: %s", err, out)
	}
}
