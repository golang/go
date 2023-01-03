// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package coverage

import (
	"fmt"
	"internal/coverage"
	"internal/goexperiment"
	"internal/platform"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Set to true for debugging (linux only).
const fixedTestDir = false

func TestCoverageApis(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping test: too long for short mode")
	}
	if !goexperiment.CoverageRedesign {
		t.Skipf("skipping new coverage tests (experiment not enabled)")
	}
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	if fixedTestDir {
		dir = "/tmp/qqqzzz"
		os.RemoveAll(dir)
		mkdir(t, dir)
	}

	// Build harness.
	bdir := mkdir(t, filepath.Join(dir, "build"))
	hargs := []string{"-cover", "-coverpkg=all"}
	if testing.CoverMode() != "" {
		hargs = append(hargs, "-covermode="+testing.CoverMode())
	}
	harnessPath := buildHarness(t, bdir, hargs)

	t.Logf("harness path is %s", harnessPath)

	// Sub-tests for each API we want to inspect, plus
	// extras for error testing.
	t.Run("emitToDir", func(t *testing.T) {
		t.Parallel()
		testEmitToDir(t, harnessPath, dir)
	})
	t.Run("emitToWriter", func(t *testing.T) {
		t.Parallel()
		testEmitToWriter(t, harnessPath, dir)
	})
	t.Run("emitToNonexistentDir", func(t *testing.T) {
		t.Parallel()
		testEmitToNonexistentDir(t, harnessPath, dir)
	})
	t.Run("emitToNilWriter", func(t *testing.T) {
		t.Parallel()
		testEmitToNilWriter(t, harnessPath, dir)
	})
	t.Run("emitToFailingWriter", func(t *testing.T) {
		t.Parallel()
		testEmitToFailingWriter(t, harnessPath, dir)
	})
	t.Run("emitWithCounterClear", func(t *testing.T) {
		t.Parallel()
		testEmitWithCounterClear(t, harnessPath, dir)
	})

}

// upmergeCoverData helps improve coverage data for this package
// itself. If this test itself is being invoked with "-cover", then
// what we'd like is for package coverage data (that is, coverage for
// routines in "runtime/coverage") to be incorporated into the test
// run from the "harness.exe" runs we've just done. We can accomplish
// this by doing a merge from the harness gocoverdir's to the test
// gocoverdir.
func upmergeCoverData(t *testing.T, gocoverdir string) {
	if testing.CoverMode() == "" {
		return
	}
	testGoCoverDir := os.Getenv("GOCOVERDIR")
	if testGoCoverDir == "" {
		return
	}
	args := []string{"tool", "covdata", "merge", "-pkg=runtime/coverage",
		"-o", testGoCoverDir, "-i", gocoverdir}
	t.Logf("up-merge of covdata from %s to %s", gocoverdir, testGoCoverDir)
	t.Logf("executing: go %+v", args)
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("covdata merge failed (%v): %s", err, b)
	}
}

// buildHarness builds the helper program "harness.exe".
func buildHarness(t *testing.T, dir string, opts []string) string {
	harnessPath := filepath.Join(dir, "harness.exe")
	harnessSrc := filepath.Join("testdata", "harness.go")
	args := []string{"build", "-o", harnessPath}
	args = append(args, opts...)
	args = append(args, harnessSrc)
	//t.Logf("harness build: go %+v\n", args)
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed (%v): %s", err, b)
	}
	return harnessPath
}

func mkdir(t *testing.T, d string) string {
	t.Helper()
	if err := os.Mkdir(d, 0777); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	return d
}

// updateGoCoverDir updates the specified environment 'env' to set
// GOCOVERDIR to 'gcd' (if setGoCoverDir is TRUE) or removes
// GOCOVERDIR from the environment (if setGoCoverDir is false).
func updateGoCoverDir(env []string, gcd string, setGoCoverDir bool) []string {
	rv := []string{}
	found := false
	for _, v := range env {
		if strings.HasPrefix(v, "GOCOVERDIR=") {
			if !setGoCoverDir {
				continue
			}
			v = "GOCOVERDIR=" + gcd
			found = true
		}
		rv = append(rv, v)
	}
	if !found && setGoCoverDir {
		rv = append(rv, "GOCOVERDIR="+gcd)
	}
	return rv
}

func runHarness(t *testing.T, harnessPath string, tp string, setGoCoverDir bool, rdir, edir string) (string, error) {
	t.Logf("running: %s -tp %s -o %s with rdir=%s and GOCOVERDIR=%v", harnessPath, tp, edir, rdir, setGoCoverDir)
	cmd := exec.Command(harnessPath, "-tp", tp, "-o", edir)
	cmd.Dir = rdir
	cmd.Env = updateGoCoverDir(os.Environ(), rdir, setGoCoverDir)
	b, err := cmd.CombinedOutput()
	//t.Logf("harness run output: %s\n", string(b))
	return string(b), err
}

func testForSpecificFunctions(t *testing.T, dir string, want []string, avoid []string) string {
	args := []string{"tool", "covdata", "debugdump",
		"-live", "-pkg=command-line-arguments", "-i=" + dir}
	t.Logf("running: go %v\n", args)
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("'go tool covdata failed (%v): %s", err, b)
	}
	output := string(b)
	rval := ""
	for _, f := range want {
		wf := "Func: " + f + "\n"
		if strings.Contains(output, wf) {
			continue
		}
		rval += fmt.Sprintf("error: output should contain %q but does not\n", wf)
	}
	for _, f := range avoid {
		wf := "Func: " + f + "\n"
		if strings.Contains(output, wf) {
			rval += fmt.Sprintf("error: output should not contain %q but does\n", wf)
		}
	}
	if rval != "" {
		t.Logf("=-= begin output:\n" + output + "\n=-= end output\n")
	}
	return rval
}

func withAndWithoutRunner(f func(setit bool, tag string)) {
	// Run 'f' with and without GOCOVERDIR set.
	for i := 0; i < 2; i++ {
		tag := "x"
		setGoCoverDir := true
		if i == 0 {
			setGoCoverDir = false
			tag = "y"
		}
		f(setGoCoverDir, tag)
	}
}

func mktestdirs(t *testing.T, tag, tp, dir string) (string, string) {
	t.Helper()
	rdir := mkdir(t, filepath.Join(dir, tp+"-rdir-"+tag))
	edir := mkdir(t, filepath.Join(dir, tp+"-edir-"+tag))
	return rdir, edir
}

func testEmitToDir(t *testing.T, harnessPath string, dir string) {
	withAndWithoutRunner(func(setGoCoverDir bool, tag string) {
		tp := "emitToDir"
		rdir, edir := mktestdirs(t, tag, tp, dir)
		output, err := runHarness(t, harnessPath, tp,
			setGoCoverDir, rdir, edir)
		if err != nil {
			t.Logf("%s", output)
			t.Fatalf("running 'harness -tp emitDir': %v", err)
		}

		// Just check to make sure meta-data file and counter data file were
		// written. Another alternative would be to run "go tool covdata"
		// or equivalent, but for now, this is what we've got.
		dents, err := os.ReadDir(edir)
		if err != nil {
			t.Fatalf("os.ReadDir(%s) failed: %v", edir, err)
		}
		mfc := 0
		cdc := 0
		for _, e := range dents {
			if e.IsDir() {
				continue
			}
			if strings.HasPrefix(e.Name(), coverage.MetaFilePref) {
				mfc++
			} else if strings.HasPrefix(e.Name(), coverage.CounterFilePref) {
				cdc++
			}
		}
		wantmf := 1
		wantcf := 1
		if mfc != wantmf {
			t.Errorf("EmitToDir: want %d meta-data files, got %d\n", wantmf, mfc)
		}
		if cdc != wantcf {
			t.Errorf("EmitToDir: want %d counter-data files, got %d\n", wantcf, cdc)
		}
		upmergeCoverData(t, edir)
		upmergeCoverData(t, rdir)
	})
}

func testEmitToWriter(t *testing.T, harnessPath string, dir string) {
	withAndWithoutRunner(func(setGoCoverDir bool, tag string) {
		tp := "emitToWriter"
		rdir, edir := mktestdirs(t, tag, tp, dir)
		output, err := runHarness(t, harnessPath, tp, setGoCoverDir, rdir, edir)
		if err != nil {
			t.Logf("%s", output)
			t.Fatalf("running 'harness -tp %s': %v", tp, err)
		}
		want := []string{"main", tp}
		avoid := []string{"final"}
		if msg := testForSpecificFunctions(t, edir, want, avoid); msg != "" {
			t.Errorf("coverage data from %q output match failed: %s", tp, msg)
		}
		upmergeCoverData(t, edir)
		upmergeCoverData(t, rdir)
	})
}

func testEmitToNonexistentDir(t *testing.T, harnessPath string, dir string) {
	withAndWithoutRunner(func(setGoCoverDir bool, tag string) {
		tp := "emitToNonexistentDir"
		rdir, edir := mktestdirs(t, tag, tp, dir)
		output, err := runHarness(t, harnessPath, tp, setGoCoverDir, rdir, edir)
		if err != nil {
			t.Logf("%s", output)
			t.Fatalf("running 'harness -tp %s': %v", tp, err)
		}
		upmergeCoverData(t, edir)
		upmergeCoverData(t, rdir)
	})
}

func testEmitToUnwritableDir(t *testing.T, harnessPath string, dir string) {
	withAndWithoutRunner(func(setGoCoverDir bool, tag string) {

		tp := "emitToUnwritableDir"
		rdir, edir := mktestdirs(t, tag, tp, dir)

		// Make edir unwritable.
		if err := os.Chmod(edir, 0555); err != nil {
			t.Fatalf("chmod failed: %v", err)
		}
		defer os.Chmod(edir, 0777)

		output, err := runHarness(t, harnessPath, tp, setGoCoverDir, rdir, edir)
		if err != nil {
			t.Logf("%s", output)
			t.Fatalf("running 'harness -tp %s': %v", tp, err)
		}
		upmergeCoverData(t, edir)
		upmergeCoverData(t, rdir)
	})
}

func testEmitToNilWriter(t *testing.T, harnessPath string, dir string) {
	withAndWithoutRunner(func(setGoCoverDir bool, tag string) {
		tp := "emitToNilWriter"
		rdir, edir := mktestdirs(t, tag, tp, dir)
		output, err := runHarness(t, harnessPath, tp, setGoCoverDir, rdir, edir)
		if err != nil {
			t.Logf("%s", output)
			t.Fatalf("running 'harness -tp %s': %v", tp, err)
		}
		upmergeCoverData(t, edir)
		upmergeCoverData(t, rdir)
	})
}

func testEmitToFailingWriter(t *testing.T, harnessPath string, dir string) {
	withAndWithoutRunner(func(setGoCoverDir bool, tag string) {
		tp := "emitToFailingWriter"
		rdir, edir := mktestdirs(t, tag, tp, dir)
		output, err := runHarness(t, harnessPath, tp, setGoCoverDir, rdir, edir)
		if err != nil {
			t.Logf("%s", output)
			t.Fatalf("running 'harness -tp %s': %v", tp, err)
		}
		upmergeCoverData(t, edir)
		upmergeCoverData(t, rdir)
	})
}

func testEmitWithCounterClear(t *testing.T, harnessPath string, dir string) {
	// Ensure that we have two versions of the harness: one built with
	// -covermode=atomic and one built with -covermode=set (we need
	// both modes to test all of the functionality).
	var nonatomicHarnessPath, atomicHarnessPath string
	if testing.CoverMode() != "atomic" {
		nonatomicHarnessPath = harnessPath
		bdir2 := mkdir(t, filepath.Join(dir, "build2"))
		hargs := []string{"-covermode=atomic", "-coverpkg=all"}
		atomicHarnessPath = buildHarness(t, bdir2, hargs)
	} else {
		atomicHarnessPath = harnessPath
		mode := "set"
		if testing.CoverMode() != "" && testing.CoverMode() != "atomic" {
			mode = testing.CoverMode()
		}
		// Build a special nonatomic covermode version of the harness
		// (we need both modes to test all of the functionality).
		bdir2 := mkdir(t, filepath.Join(dir, "build2"))
		hargs := []string{"-covermode=" + mode, "-coverpkg=all"}
		nonatomicHarnessPath = buildHarness(t, bdir2, hargs)
	}

	withAndWithoutRunner(func(setGoCoverDir bool, tag string) {
		// First a run with the nonatomic harness path, which we
		// expect to fail.
		tp := "emitWithCounterClear"
		rdir1, edir1 := mktestdirs(t, tag, tp+"1", dir)
		output, err := runHarness(t, nonatomicHarnessPath, tp,
			setGoCoverDir, rdir1, edir1)
		if err == nil {
			t.Logf("%s", output)
			t.Fatalf("running '%s -tp %s': unexpected success",
				nonatomicHarnessPath, tp)
		}

		// Next a run with the atomic harness path, which we
		// expect to succeed.
		rdir2, edir2 := mktestdirs(t, tag, tp+"2", dir)
		output, err = runHarness(t, atomicHarnessPath, tp,
			setGoCoverDir, rdir2, edir2)
		if err != nil {
			t.Logf("%s", output)
			t.Fatalf("running 'harness -tp %s': %v", tp, err)
		}
		want := []string{tp, "postClear"}
		avoid := []string{"preClear", "main", "final"}
		if msg := testForSpecificFunctions(t, edir2, want, avoid); msg != "" {
			t.Logf("%s", output)
			t.Errorf("coverage data from %q output match failed: %s", tp, msg)
		}

		if testing.CoverMode() == "atomic" {
			upmergeCoverData(t, edir2)
			upmergeCoverData(t, rdir2)
		} else {
			upmergeCoverData(t, edir1)
			upmergeCoverData(t, rdir1)
		}
	})
}

func TestApisOnNocoverBinary(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping test: too long for short mode")
	}
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()

	// Build harness with no -cover.
	bdir := mkdir(t, filepath.Join(dir, "nocover"))
	edir := mkdir(t, filepath.Join(dir, "emitDirNo"))
	harnessPath := buildHarness(t, bdir, nil)
	output, err := runHarness(t, harnessPath, "emitToDir", false, edir, edir)
	if err == nil {
		t.Fatalf("expected error on TestApisOnNocoverBinary harness run")
	}
	const want = "not built with -cover"
	if !strings.Contains(output, want) {
		t.Errorf("error output does not contain %q: %s", want, output)
	}
}

func TestIssue56006EmitDataRaceCoverRunningGoroutine(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping test: too long for short mode")
	}
	if !goexperiment.CoverageRedesign {
		t.Skipf("skipping new coverage tests (experiment not enabled)")
	}

	// This test requires "go test -race -cover", meaning that we need
	// go build, go run, and "-race" support.
	testenv.MustHaveGoRun(t)
	if !platform.RaceDetectorSupported(runtime.GOOS, runtime.GOARCH) ||
		!testenv.HasCGO() {
		t.Skip("skipped due to lack of race detector support / CGO")
	}

	// This will run a program with -cover and -race where we have a
	// goroutine still running (and updating counters) at the point where
	// the test runtime is trying to write out counter data.
	cmd := exec.Command(testenv.GoToolPath(t), "test", "-cover", "-race")
	cmd.Dir = filepath.Join("testdata", "issue56006")
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go test -cover -race failed: %v", err)
	}

	// Don't want to see any data races in output.
	avoid := []string{"DATA RACE"}
	for _, no := range avoid {
		if strings.Contains(string(b), no) {
			t.Logf("%s\n", string(b))
			t.Fatalf("found %s in test output, not permitted", no)
		}
	}
}
