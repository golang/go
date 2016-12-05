// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"internal/testenv"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TODO: move all these tests elsewhere?
// Perhaps teach test/run.go how to run them with a new action verb.
func runTest(t *testing.T, filename string) {
	t.Parallel()
	doTest(t, filename, "run")
}
func buildTest(t *testing.T, filename string) {
	t.Parallel()
	doTest(t, filename, "build")
}
func doTest(t *testing.T, filename string, kind string) {
	testenv.MustHaveGoBuild(t)
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(testenv.GoToolPath(t), kind, filepath.Join("testdata", filename))
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed: %v:\nOut: %s\nStderr: %s\n", err, &stdout, &stderr)
	}
	if s := stdout.String(); s != "" {
		t.Errorf("Stdout = %s\nWant empty", s)
	}
	if s := stderr.String(); strings.Contains(s, "SSA unimplemented") {
		t.Errorf("Unimplemented message found in stderr:\n%s", s)
	}
}

// TestShortCircuit tests OANDAND and OOROR expressions and short circuiting.
func TestShortCircuit(t *testing.T) { runTest(t, "short.go") }

// TestBreakContinue tests that continue and break statements do what they say.
func TestBreakContinue(t *testing.T) { runTest(t, "break.go") }

// TestTypeAssertion tests type assertions.
func TestTypeAssertion(t *testing.T) { runTest(t, "assert.go") }

// TestArithmetic tests that both backends have the same result for arithmetic expressions.
func TestArithmetic(t *testing.T) { runTest(t, "arith.go") }

// TestFP tests that both backends have the same result for floating point expressions.
func TestFP(t *testing.T) { runTest(t, "fp.go") }

// TestArithmeticBoundary tests boundary results for arithmetic operations.
func TestArithmeticBoundary(t *testing.T) { runTest(t, "arithBoundary.go") }

// TestArithmeticConst tests results for arithmetic operations against constants.
func TestArithmeticConst(t *testing.T) { runTest(t, "arithConst.go") }

func TestChan(t *testing.T) { runTest(t, "chan.go") }

func TestCompound(t *testing.T) { runTest(t, "compound.go") }

func TestCtl(t *testing.T) { runTest(t, "ctl.go") }

func TestLoadStore(t *testing.T) { runTest(t, "loadstore.go") }

func TestMap(t *testing.T) { runTest(t, "map.go") }

func TestRegalloc(t *testing.T) { runTest(t, "regalloc.go") }

func TestString(t *testing.T) { runTest(t, "string.go") }

func TestDeferNoReturn(t *testing.T) { buildTest(t, "deferNoReturn.go") }

// TestClosure tests closure related behavior.
func TestClosure(t *testing.T) { runTest(t, "closure.go") }

func TestArray(t *testing.T) { runTest(t, "array.go") }

func TestAppend(t *testing.T) { runTest(t, "append.go") }

func TestZero(t *testing.T) { runTest(t, "zero.go") }

func TestAddressed(t *testing.T) { runTest(t, "addressed.go") }

func TestCopy(t *testing.T) { runTest(t, "copy.go") }

func TestUnsafe(t *testing.T) { runTest(t, "unsafe.go") }

func TestPhi(t *testing.T) { runTest(t, "phi.go") }

func TestSlice(t *testing.T) { runTest(t, "slice.go") }

func TestNamedReturn(t *testing.T) { runTest(t, "namedReturn.go") }

func TestDuplicateLoad(t *testing.T) { runTest(t, "dupLoad.go") }

func TestSqrt(t *testing.T) { runTest(t, "sqrt_const.go") }
