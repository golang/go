// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestLargeBranch generates a large function with a very far conditional
// branch, in order to ensure that it assembles successfully.
func TestLargeBranch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()

	// Generate a very large function.
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	genLargeBranch(buf)

	tmpfile := filepath.Join(dir, "x.s")
	if err := os.WriteFile(tmpfile, buf.Bytes(), 0644); err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	// Assemble generated file.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
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

	dir := t.TempDir()

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module largecall"), 0644); err != nil {
		t.Fatalf("Failed to write file: %v\n", err)
	}
	main := `package main
func main() {
        x()
}

func x()
func y()
`
	if err := os.WriteFile(filepath.Join(dir, "x.go"), []byte(main), 0644); err != nil {
		t.Fatalf("failed to write main: %v\n", err)
	}

	// Generate a very large function with call.
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	genLargeCall(buf)

	if err := os.WriteFile(filepath.Join(dir, "x.s"), buf.Bytes(), 0644); err != nil {
		t.Fatalf("Failed to write file: %v\n", err)
	}

	// Build generated files.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-linkmode=internal")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}

	if runtime.GOARCH == "riscv64" && testenv.HasCGO() {
		cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-linkmode=external")
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

// TestLargeJump generates a large jump (>1MB of text) with a JMP to the
// end of the function, in order to ensure that it assembles correctly.
func TestLargeJump(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	if runtime.GOARCH != "riscv64" {
		t.Skip("Require riscv64 to run")
	}
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module largejump"), 0644); err != nil {
		t.Fatalf("Failed to write file: %v\n", err)
	}
	main := `package main

import "fmt"

func main() {
        fmt.Print(x())
}

func x() uint64
`
	if err := os.WriteFile(filepath.Join(dir, "x.go"), []byte(main), 0644); err != nil {
		t.Fatalf("failed to write main: %v\n", err)
	}

	// Generate a very large jump instruction.
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	genLargeJump(buf)

	if err := os.WriteFile(filepath.Join(dir, "x.s"), buf.Bytes(), 0644); err != nil {
		t.Fatalf("Failed to write file: %v\n", err)
	}

	// Build generated files.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", "x.exe")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}

	cmd = testenv.Command(t, filepath.Join(dir, "x.exe"))
	out, err = cmd.CombinedOutput()
	if string(out) != "1" {
		t.Errorf(`Got test output %q, want "1"`, string(out))
	}
}

func genLargeJump(buf *bytes.Buffer) {
	fmt.Fprintln(buf, "TEXT ·x(SB),0,$0-8")
	fmt.Fprintln(buf, "MOV  X0, X10")
	fmt.Fprintln(buf, "JMP end")
	for i := 0; i < 1<<18; i++ {
		fmt.Fprintln(buf, "ADD $1, X10, X10")
	}
	fmt.Fprintln(buf, "end:")
	fmt.Fprintln(buf, "ADD $1, X10, X10")
	fmt.Fprintln(buf, "MOV X10, r+0(FP)")
	fmt.Fprintln(buf, "RET")
}

// Issue 20348.
func TestNoRet(t *testing.T) {
	dir := t.TempDir()
	tmpfile := filepath.Join(dir, "x.s")
	if err := os.WriteFile(tmpfile, []byte("TEXT ·stub(SB),$0-0\nNOP\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("%v\n%s", err, out)
	}
}

func TestImmediateSplitting(t *testing.T) {
	dir := t.TempDir()
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
	if err := os.WriteFile(tmpfile, []byte(asm), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
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

	cmd := testenv.Command(t, testenv.GoToolPath(t), "test")
	cmd.Dir = "testdata/testbranch"
	if out, err := testenv.CleanCmdEnv(cmd).CombinedOutput(); err != nil {
		t.Errorf("Branch test failed: %v\n%s", err, out)
	}
}

func TestPCAlign(t *testing.T) {
	dir := t.TempDir()
	tmpfile := filepath.Join(dir, "x.s")
	asm := `
TEXT _stub(SB),$0-0
	FENCE
	PCALIGN	$8
	FENCE
	RET
`
	if err := os.WriteFile(tmpfile, []byte(asm), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), "-S", tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=riscv64", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Failed to assemble: %v\n%s", err, out)
	}
	// The expected instruction sequence after alignment:
	//	FENCE
	//	NOP
	//	FENCE
	//	RET
	want := "0f 00 f0 0f 13 00 00 00 0f 00 f0 0f 67 80 00 00"
	if !strings.Contains(string(out), want) {
		t.Errorf("PCALIGN test failed - got %s\nwant %s", out, want)
	}
}
