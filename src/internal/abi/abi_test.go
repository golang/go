// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi_test

import (
	"internal/abi"
	"internal/testenv"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFuncPC(t *testing.T) {
	// Test that FuncPC* can get correct function PC.
	pcFromAsm := abi.FuncPCTestFnAddr

	// Test FuncPC for locally defined function
	pcFromGo := abi.FuncPCTest()
	if pcFromGo != pcFromAsm {
		t.Errorf("FuncPC returns wrong PC, want %x, got %x", pcFromAsm, pcFromGo)
	}

	// Test FuncPC for imported function
	pcFromGo = abi.FuncPCABI0(abi.FuncPCTestFn)
	if pcFromGo != pcFromAsm {
		t.Errorf("FuncPC returns wrong PC, want %x, got %x", pcFromAsm, pcFromGo)
	}
}

func TestFuncPCCompileError(t *testing.T) {
	// Test that FuncPC* on a function of a mismatched ABI is rejected.
	testenv.MustHaveGoBuild(t)

	// We want to test internal package, which we cannot normally import.
	// Run the assembler and compiler manually.
	tmpdir := t.TempDir()
	asmSrc := filepath.Join("testdata", "x.s")
	goSrc := filepath.Join("testdata", "x.go")
	symabi := filepath.Join(tmpdir, "symabi")
	obj := filepath.Join(tmpdir, "x.o")

	// parse assembly code for symabi.
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-gensymabis", "-o", symabi, asmSrc)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go tool asm -gensymabis failed: %v\n%s", err, out)
	}

	// compile go code.
	cmd = exec.Command(testenv.GoToolPath(t), "tool", "compile", "-symabis", symabi, "-o", obj, goSrc)
	out, err = cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("go tool compile did not fail")
	}

	// Expect errors in line 17, 18, 20, no errors on other lines.
	want := []string{"x.go:17", "x.go:18", "x.go:20"}
	got := strings.Split(string(out), "\n")
	if got[len(got)-1] == "" {
		got = got[:len(got)-1] // remove last empty line
	}
	for i, s := range got {
		if !strings.Contains(s, want[i]) {
			t.Errorf("did not error on line %s", want[i])
		}
	}
	if len(got) != len(want) {
		t.Errorf("unexpected number of errors, want %d, got %d", len(want), len(got))
	}
	if t.Failed() {
		t.Logf("output:\n%s", string(out))
	}
}
