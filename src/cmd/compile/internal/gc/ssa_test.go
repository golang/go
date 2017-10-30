// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os/exec"
	"path/filepath"
	"runtime"
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

// runGenTest runs a test-generator, then runs the generated test.
// Generated test can either fail in compilation or execution.
// The environment variable parameter(s) is passed to the run
// of the generated test.
func runGenTest(t *testing.T, filename, tmpname string, ev ...string) {
	testenv.MustHaveGoRun(t)
	var stdout, stderr bytes.Buffer
	cmd := exec.Command("go", "run", filepath.Join("testdata", filename))
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed: %v:\nOut: %s\nStderr: %s\n", err, &stdout, &stderr)
	}
	// Write stdout into a temporary file
	tmpdir, ok := ioutil.TempDir("", tmpname)
	if ok != nil {
		t.Fatalf("Failed to create temporary directory")
	}

	rungo := filepath.Join(tmpdir, "run.go")
	ok = ioutil.WriteFile(rungo, stdout.Bytes(), 0600)
	if ok != nil {
		t.Fatalf("Failed to create temporary file " + rungo)
	}

	stdout.Reset()
	stderr.Reset()
	// Execute compile+link+run instead of "go run" to avoid applying -gcflags=-d=ssa/check/on
	// to the runtime (especially over and over and over).
	// compile
	cmd = exec.Command("go", "tool", "compile", "-d=ssa/check/on", "-o", filepath.Join(tmpdir, "run.a"), rungo)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Env = append(cmd.Env, ev...)
	err := cmd.Run()
	if err == nil {
		// link
		cmd = exec.Command("go", "tool", "link", "-o", filepath.Join(tmpdir, "run.exe"), filepath.Join(tmpdir, "run.a"))
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		cmd.Env = append(cmd.Env, ev...)
		err = cmd.Run()
	}
	if err == nil {
		// run
		cmd = exec.Command(filepath.Join(tmpdir, "run.exe"))
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		cmd.Env = append(cmd.Env, ev...)
		err = cmd.Run()
	}
	if err != nil {
		t.Fatalf("Failed: %v:\nOut: %s\nStderr: %s\n", err, &stdout, &stderr)
	}
	if s := stderr.String(); s != "" {
		t.Errorf("Stderr = %s\nWant empty", s)
	}
	if s := stdout.String(); s != "" {
		t.Errorf("Stdout = %s\nWant empty", s)
	}
}

func TestGenFlowGraph(t *testing.T) {
	runGenTest(t, "flowgraph_generator1.go", "ssa_fg_tmp1")
	if runtime.GOOS != "windows" {
		runGenTest(t, "flowgraph_generator1.go", "ssa_fg_tmp2", "GO_SSA_PHI_LOC_CUTOFF=0")
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

// TestComparisonsConst tests results for comparison operations against constants.
func TestComparisonsConst(t *testing.T) { runTest(t, "cmpConst.go") }

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
