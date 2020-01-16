// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestLarge generates a very large file to verify that large
// program builds successfully, in particular, too-far
// conditional branches are fixed.
func TestLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("Skip in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir, err := ioutil.TempDir("", "testlarge")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Generate a very large function.
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	gen(buf)

	tmpfile := filepath.Join(dir, "x.s")
	err = ioutil.WriteFile(tmpfile, buf.Bytes(), 0644)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	// Build generated file.
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}
}

// gen generates a very large program, with a very far conditional branch.
func gen(buf *bytes.Buffer) {
	fmt.Fprintln(buf, "TEXT f(SB),0,$0-0")
	fmt.Fprintln(buf, "BEQ X0, X0, label")
	for i := 0; i < 1<<19; i++ {
		fmt.Fprintln(buf, "ADD $0, X0, X0")
	}
	fmt.Fprintln(buf, "label:")
	fmt.Fprintln(buf, "ADD $0, X0, X0")
}

// Issue 20348.
func TestNoRet(t *testing.T) {
	dir, err := ioutil.TempDir("", "testnoret")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	tmpfile := filepath.Join(dir, "x.s")
	if err := ioutil.WriteFile(tmpfile, []byte("TEXT ·stub(SB),$0-0\nNOP\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("%v\n%s", err, out)
	}
}

func TestImmediateSplitting(t *testing.T) {
	dir, err := ioutil.TempDir("", "testimmsplit")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	tmpfile := filepath.Join(dir, "x.s")
	asm := `
TEXT _stub(SB),$0-0
	LB	4096(X5), X6
	LH	4096(X5), X6
	LW	4096(X5), X6
	LD	4096(X5), X6
	LBU	4096(X5), X6
	LHU	4096(X5), X6
	LWU	4096(X5), X6
	SB	X6, 4096(X5)
	SH	X6, 4096(X5)
	SW	X6, 4096(X5)
	SD	X6, 4096(X5)

	FLW	4096(X5), F6
	FLD	4096(X5), F6
	FSW	F6, 4096(X5)
	FSD	F6, 4096(X5)

	MOVB	4096(X5), X6
	MOVH	4096(X5), X6
	MOVW	4096(X5), X6
	MOV	4096(X5), X6
	MOVBU	4096(X5), X6
	MOVHU	4096(X5), X6
	MOVWU	4096(X5), X6

	MOVB	X6, 4096(X5)
	MOVH	X6, 4096(X5)
	MOVW	X6, 4096(X5)
	MOV	X6, 4096(X5)

	MOVF	4096(X5), F6
	MOVD	4096(X5), F6
	MOVF	F6, 4096(X5)
	MOVD	F6, 4096(X5)
`
	if err := ioutil.WriteFile(tmpfile, []byte(asm), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("%v\n%s", err, out)
	}
}
