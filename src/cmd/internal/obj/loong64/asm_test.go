// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"testing"
)

const genBufSize = (1024 * 1024 * 32) // 32MB

// TestLargeBranch generates a large function with a very far conditional
// branch, in order to ensure that it assembles successfully.
func TestLargeBranch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir, err := os.MkdirTemp("", "testlargebranch")
	if err != nil {
		t.Fatalf("Could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Generate a very large function.
	buf := bytes.NewBuffer(make([]byte, 0, genBufSize))
	genLargeBranch(buf)

	tmpfile := filepath.Join(dir, "x.s")
	if err := os.WriteFile(tmpfile, buf.Bytes(), 0644); err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	// Assemble generated file.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=loong64", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}
}

func genLargeBranch(buf *bytes.Buffer) {
	genSize1 := (1 << 16) + 16
	genSize2 := (1 << 21) + 16

	fmt.Fprintln(buf, "TEXT f(SB),0,$0-0")
	fmt.Fprintln(buf, "BEQ R5, R6, label18")
	fmt.Fprintln(buf, "BNE R5, R6, label18")
	fmt.Fprintln(buf, "BGE R5, R6, label18")

	fmt.Fprintln(buf, "BGEU R5, R6, label18")
	fmt.Fprintln(buf, "BLTU R5, R6, label18")

	fmt.Fprintln(buf, "BLEZ R5, label18")
	fmt.Fprintln(buf, "BGEZ R5, label18")
	fmt.Fprintln(buf, "BLTZ R5, label18")
	fmt.Fprintln(buf, "BGTZ R5, label18")

	fmt.Fprintln(buf, "BFPT label23")
	fmt.Fprintln(buf, "BFPF label23")

	fmt.Fprintln(buf, "BEQ R5, label23")
	fmt.Fprintln(buf, "BNE R5, label23")

	for i := 0; i <= genSize1; i++ {
		fmt.Fprintln(buf, "ADDV $0, R0, R0")
	}

	fmt.Fprintln(buf, "label18:")
	for i := 0; i <= (genSize2 - genSize1); i++ {
		fmt.Fprintln(buf, "ADDV $0, R0, R0")
	}

	fmt.Fprintln(buf, "label23:")
	fmt.Fprintln(buf, "ADDV $0, R0, R0")
	fmt.Fprintln(buf, "RET")
}

// TestPCALIGN verifies the correctness of the PCALIGN by checking if the
// code can be aligned to the alignment value.
func TestPCALIGN(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	tmpfile := filepath.Join(dir, "testpcalign.s")
	tmpout := filepath.Join(dir, "testpcalign.o")

	code1 := []byte("TEXT ·foo(SB),$0-0\nMOVW $0, R0\nPCALIGN $8\nADDV $8, R0\nRET\n")
	code2 := []byte("TEXT ·foo(SB),$0-0\nMOVW $0, R0\nPCALIGN $16\nADDV $16, R0\nRET\n")
	code3 := []byte("TEXT ·foo(SB),$0-0\nMOVW $0, R0\nPCALIGN $32\nADDV $32, R0\nRET\n")
	out1 := `0x0008\s00008\s\(.*\)\s*ADDV\s\$8,\sR0`
	out2 := `0x0010\s00016\s\(.*\)\s*ADDV\s\$16,\sR0`
	out3 := `0x0020\s00032\s\(.*\)\s*ADDV\s\$32,\sR0`
	var testCases = []struct {
		name   string
		source []byte
		want   string
	}{
		{"pcalign8", code1, out1},
		{"pcalign16", code2, out2},
		{"pcalign32", code3, out3},
	}
	for _, test := range testCases {
		if err := os.WriteFile(tmpfile, test.source, 0644); err != nil {
			t.Fatal(err)
		}
		cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-S", "-o", tmpout, tmpfile)
		cmd.Env = append(os.Environ(), "GOARCH=loong64", "GOOS=linux")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("The %s build failed: %v, output: %s", test.name, err, out)
			continue
		}
		matched, err := regexp.MatchString(test.want, string(out))
		if err != nil {
			t.Fatal(err)
		}
		if !matched {
			t.Errorf("The %s testing failed!\ninput: %s\noutput: %s\n", test.name, test.source, out)
		}
	}
}

func TestNoRet(t *testing.T) {
	dir := t.TempDir()
	tmpfile := filepath.Join(dir, "testnoret.s")
	tmpout := filepath.Join(dir, "testnoret.o")
	if err := os.WriteFile(tmpfile, []byte("TEXT ·foo(SB),$0-0\nNOP\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", tmpout, tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=loong64", "GOOS=linux")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("%v\n%s", err, out)
	}
}

func TestLargeCall(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	if runtime.GOARCH != "loong64" {
		t.Skip("Require loong64 to run")
	}
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module largecall"), 0644); err != nil {
		t.Fatalf("Failed to write file: %v\n", err)
	}
	main := `package main

func main() {
        a()
}

func a()
`
	if err := os.WriteFile(filepath.Join(dir, "largecall.go"), []byte(main), 0644); err != nil {
		t.Fatalf("failed to write main: %v\n", err)
	}

	// Generate a very large call instruction.
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	genLargeCall(buf)

	if err := os.WriteFile(filepath.Join(dir, "largecall.s"), buf.Bytes(), 0644); err != nil {
		t.Fatalf("Failed to write file: %v\n", err)
	}

	// Build generated files.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}
}

func genLargeCall(buf *bytes.Buffer) {
	fmt.Fprintln(buf, "TEXT main·a(SB),0,$0-8")
	fmt.Fprintln(buf, "CALL b(SB)")
	for i := 0; i <= ((1 << 26) + 26); i++ {
		fmt.Fprintln(buf, "ADDV $0, R0, R0")
	}
	fmt.Fprintln(buf, "RET")
	fmt.Fprintln(buf, "TEXT b(SB),0,$0-8")
	fmt.Fprintln(buf, "ADDV $0, R0, R0")
	fmt.Fprintln(buf, "RET")
}
