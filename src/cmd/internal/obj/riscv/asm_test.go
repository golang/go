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
	"runtime"
	"testing"
)

// TestLargeBranch generates a large function with a very far conditional
// branch, in order to ensure that it assembles successfully.
func TestLargeBranch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir, err := ioutil.TempDir("", "testlargebranch")
	if err != nil {
		t.Fatalf("Could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Generate a very large function.
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	genLargeBranch(buf)

	tmpfile := filepath.Join(dir, "x.s")
	if err := ioutil.WriteFile(tmpfile, buf.Bytes(), 0644); err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	// Assemble generated file.
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}
}

func genLargeBranch(buf *bytes.Buffer) {
	fmt.Fprintln(buf, "TEXT f(SB),0,$0-0")
	fmt.Fprintln(buf, "BEQ X0, X0, label")
	for i := 0; i < 1<<19; i++ {
		fmt.Fprintln(buf, "ADD $0, X0, X0")
	}
	fmt.Fprintln(buf, "label:")
	fmt.Fprintln(buf, "ADD $0, X0, X0")
}

// TestLargeCall generates a large function (>1MB of text) with a call to
// a following function, in order to ensure that it assembles and links
// correctly.
func TestLargeCall(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir, err := ioutil.TempDir("", "testlargecall")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	if err := ioutil.WriteFile(filepath.Join(dir, "go.mod"), []byte("module largecall"), 0644); err != nil {
		t.Fatalf("Failed to write file: %v\n", err)
	}
	main := `package main
func main() {
        x()
}

func x()
func y()
`
	if err := ioutil.WriteFile(filepath.Join(dir, "x.go"), []byte(main), 0644); err != nil {
		t.Fatalf("failed to write main: %v\n", err)
	}

	// Generate a very large function with call.
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	genLargeCall(buf)

	if err := ioutil.WriteFile(filepath.Join(dir, "x.s"), buf.Bytes(), 0644); err != nil {
		t.Fatalf("Failed to write file: %v\n", err)
	}

	// Build generated files.
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-ldflags=-linkmode=internal")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}

	if runtime.GOARCH == "riscv64" && testenv.HasCGO() {
		cmd := exec.Command(testenv.GoToolPath(t), "build", "-ldflags=-linkmode=external")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("Build failed: %v, output: %s", err, out)
		}
	}
}

func genLargeCall(buf *bytes.Buffer) {
	fmt.Fprintln(buf, "TEXT ·x(SB),0,$0-0")
	fmt.Fprintln(buf, "CALL ·y(SB)")
	for i := 0; i < 1<<19; i++ {
		fmt.Fprintln(buf, "ADD $0, X0, X0")
	}
	fmt.Fprintln(buf, "RET")
	fmt.Fprintln(buf, "TEXT ·y(SB),0,$0-0")
	fmt.Fprintln(buf, "ADD $0, X0, X0")
	fmt.Fprintln(buf, "RET")
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

func TestBranch(t *testing.T) {
	if runtime.GOARCH != "riscv64" {
		t.Skip("Requires riscv64 to run")
	}

	testenv.MustHaveGoBuild(t)

	cmd := exec.Command(testenv.GoToolPath(t), "test")
	cmd.Dir = "testdata/testbranch"
	if out, err := testenv.CleanCmdEnv(cmd).CombinedOutput(); err != nil {
		t.Errorf("Branch test failed: %v\n%s", err, out)
	}
}
