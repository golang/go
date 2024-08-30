// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"internal/race"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// This is exactly what a test would do without a TestMain.
// It's here only so that there is at least one package in the
// standard library with a TestMain, so that code is executed.

func TestMain(m *testing.M) {
	if os.Getenv("GO_WANT_RACE_BEFORE_TESTS") == "1" {
		doRace()
	}

	m.Run()

	// Note: m.Run currently prints the final "PASS" line, so if any race is
	// reported here (after m.Run but before the process exits), it will print
	// "PASS", then print the stack traces for the race, then exit with nonzero
	// status.
	//
	// This is a somewhat fundamental race: because the race detector hooks into
	// the runtime at a very low level, no matter where we put the printing it
	// would be possible to report a race that occurs afterward. However, we could
	// theoretically move the printing after TestMain, which would at least do a
	// better job of diagnosing races in cleanup functions within TestMain itself.
}

func TestTempDirInCleanup(t *testing.T) {
	var dir string

	t.Run("test", func(t *testing.T) {
		t.Cleanup(func() {
			dir = t.TempDir()
		})
		_ = t.TempDir()
	})

	fi, err := os.Stat(dir)
	if fi != nil {
		t.Fatalf("Directory %q from user Cleanup still exists", dir)
	}
	if !os.IsNotExist(err) {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestTempDirInBenchmark(t *testing.T) {
	testing.Benchmark(func(b *testing.B) {
		if !b.Run("test", func(b *testing.B) {
			// Add a loop so that the test won't fail. See issue 38677.
			for i := 0; i < b.N; i++ {
				_ = b.TempDir()
			}
		}) {
			t.Fatal("Sub test failure in a benchmark")
		}
	})
}

func TestTempDir(t *testing.T) {
	testTempDir(t)
	t.Run("InSubtest", testTempDir)
	t.Run("test/subtest", testTempDir)
	t.Run("test\\subtest", testTempDir)
	t.Run("test:subtest", testTempDir)
	t.Run("test/..", testTempDir)
	t.Run("../test", testTempDir)
	t.Run("test[]", testTempDir)
	t.Run("test*", testTempDir)
	t.Run("äöüéè", testTempDir)
}

func testTempDir(t *testing.T) {
	dirCh := make(chan string, 1)
	t.Cleanup(func() {
		// Verify directory has been removed.
		select {
		case dir := <-dirCh:
			fi, err := os.Stat(dir)
			if os.IsNotExist(err) {
				// All good
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			t.Errorf("directory %q still exists: %v, isDir=%v", dir, fi, fi.IsDir())
		default:
			if !t.Failed() {
				t.Fatal("never received dir channel")
			}
		}
	})

	dir := t.TempDir()
	if dir == "" {
		t.Fatal("expected dir")
	}
	dir2 := t.TempDir()
	if dir == dir2 {
		t.Fatal("subsequent calls to TempDir returned the same directory")
	}
	if filepath.Dir(dir) != filepath.Dir(dir2) {
		t.Fatalf("calls to TempDir do not share a parent; got %q, %q", dir, dir2)
	}
	dirCh <- dir
	fi, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !fi.IsDir() {
		t.Errorf("dir %q is not a dir", dir)
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) > 0 {
		t.Errorf("unexpected %d files in TempDir: %v", len(files), files)
	}

	glob := filepath.Join(dir, "*.txt")
	if _, err := filepath.Glob(glob); err != nil {
		t.Error(err)
	}
}

func TestSetenv(t *testing.T) {
	tests := []struct {
		name               string
		key                string
		initialValueExists bool
		initialValue       string
		newValue           string
	}{
		{
			name:               "initial value exists",
			key:                "GO_TEST_KEY_1",
			initialValueExists: true,
			initialValue:       "111",
			newValue:           "222",
		},
		{
			name:               "initial value exists but empty",
			key:                "GO_TEST_KEY_2",
			initialValueExists: true,
			initialValue:       "",
			newValue:           "222",
		},
		{
			name:               "initial value is not exists",
			key:                "GO_TEST_KEY_3",
			initialValueExists: false,
			initialValue:       "",
			newValue:           "222",
		},
	}

	for _, test := range tests {
		if test.initialValueExists {
			if err := os.Setenv(test.key, test.initialValue); err != nil {
				t.Fatalf("unable to set env: got %v", err)
			}
		} else {
			os.Unsetenv(test.key)
		}

		t.Run(test.name, func(t *testing.T) {
			t.Setenv(test.key, test.newValue)
			if os.Getenv(test.key) != test.newValue {
				t.Fatalf("unexpected value after t.Setenv: got %s, want %s", os.Getenv(test.key), test.newValue)
			}
		})

		got, exists := os.LookupEnv(test.key)
		if got != test.initialValue {
			t.Fatalf("unexpected value after t.Setenv cleanup: got %s, want %s", got, test.initialValue)
		}
		if exists != test.initialValueExists {
			t.Fatalf("unexpected value after t.Setenv cleanup: got %t, want %t", exists, test.initialValueExists)
		}
	}
}

func expectParallelConflict(t *testing.T) {
	want := testing.ParallelConflict
	if got := recover(); got != want {
		t.Fatalf("expected panic; got %#v want %q", got, want)
	}
}

func testWithParallelAfter(t *testing.T, fn func(*testing.T)) {
	defer expectParallelConflict(t)

	fn(t)
	t.Parallel()
}

func testWithParallelBefore(t *testing.T, fn func(*testing.T)) {
	defer expectParallelConflict(t)

	t.Parallel()
	fn(t)
}

func testWithParallelParentBefore(t *testing.T, fn func(*testing.T)) {
	t.Parallel()

	t.Run("child", func(t *testing.T) {
		defer expectParallelConflict(t)

		fn(t)
	})
}

func testWithParallelGrandParentBefore(t *testing.T, fn func(*testing.T)) {
	t.Parallel()

	t.Run("child", func(t *testing.T) {
		t.Run("grand-child", func(t *testing.T) {
			defer expectParallelConflict(t)

			fn(t)
		})
	})
}

func tSetenv(t *testing.T) {
	t.Setenv("GO_TEST_KEY_1", "value")
}

func TestSetenvWithParallelAfter(t *testing.T) {
	testWithParallelAfter(t, tSetenv)
}

func TestSetenvWithParallelBefore(t *testing.T) {
	testWithParallelBefore(t, tSetenv)
}

func TestSetenvWithParallelParentBefore(t *testing.T) {
	testWithParallelParentBefore(t, tSetenv)
}

func TestSetenvWithParallelGrandParentBefore(t *testing.T) {
	testWithParallelGrandParentBefore(t, tSetenv)
}

func tChdir(t *testing.T) {
	t.Chdir(t.TempDir())
}

func TestChdirWithParallelAfter(t *testing.T) {
	testWithParallelAfter(t, tChdir)
}

func TestChdirWithParallelBefore(t *testing.T) {
	testWithParallelBefore(t, tChdir)
}

func TestChdirWithParallelParentBefore(t *testing.T) {
	testWithParallelParentBefore(t, tChdir)
}

func TestChdirWithParallelGrandParentBefore(t *testing.T) {
	testWithParallelGrandParentBefore(t, tChdir)
}

func TestChdir(t *testing.T) {
	oldDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(oldDir)

	// The "relative" test case relies on tmp not being a symlink.
	tmp, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	rel, err := filepath.Rel(oldDir, tmp)
	if err != nil {
		// when GOROOT is git clone dir,
		// there will happen error here,
		// skip this test to avoid false test failures.
		t.Skip(err)
	}

	for _, tc := range []struct {
		name, dir, pwd string
		extraChdir     bool
	}{
		{
			name: "absolute",
			dir:  tmp,
			pwd:  tmp,
		},
		{
			name: "relative",
			dir:  rel,
			pwd:  tmp,
		},
		{
			name: "current (absolute)",
			dir:  oldDir,
			pwd:  oldDir,
		},
		{
			name: "current (relative) with extra os.Chdir",
			dir:  ".",
			pwd:  oldDir,

			extraChdir: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !filepath.IsAbs(tc.pwd) {
				t.Fatalf("Bad tc.pwd: %q (must be absolute)", tc.pwd)
			}

			t.Chdir(tc.dir)

			newDir, err := os.Getwd()
			if err != nil {
				t.Fatal(err)
			}
			if newDir != tc.pwd {
				t.Fatalf("failed to chdir to %q: getwd: got %q, want %q", tc.dir, newDir, tc.pwd)
			}

			switch runtime.GOOS {
			case "windows", "plan9":
				// Windows and Plan 9 do not use the PWD variable.
			default:
				if pwd := os.Getenv("PWD"); pwd != tc.pwd {
					t.Fatalf("PWD: got %q, want %q", pwd, tc.pwd)
				}
			}

			if tc.extraChdir {
				os.Chdir("..")
			}
		})

		newDir, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		if newDir != oldDir {
			t.Fatalf("failed to restore wd to %s: getwd: %s", oldDir, newDir)
		}
	}
}

// testingTrueInInit is part of TestTesting.
var testingTrueInInit = false

// testingTrueInPackageVarInit is part of TestTesting.
var testingTrueInPackageVarInit = testing.Testing()

// init is part of TestTesting.
func init() {
	if testing.Testing() {
		testingTrueInInit = true
	}
}

var testingProg = `
package main

import (
	"fmt"
	"testing"
)

func main() {
	fmt.Println(testing.Testing())
}
`

func TestTesting(t *testing.T) {
	if !testing.Testing() {
		t.Errorf("testing.Testing() == %t, want %t", testing.Testing(), true)
	}
	if !testingTrueInInit {
		t.Errorf("testing.Testing() called by init function == %t, want %t", testingTrueInInit, true)
	}
	if !testingTrueInPackageVarInit {
		t.Errorf("testing.Testing() variable initialized as %t, want %t", testingTrueInPackageVarInit, true)
	}

	if testing.Short() {
		t.Skip("skipping building a binary in short mode")
	}
	testenv.MustHaveGoRun(t)

	fn := filepath.Join(t.TempDir(), "x.go")
	if err := os.WriteFile(fn, []byte(testingProg), 0644); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "run", fn)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v failed: %v\n%s", cmd, err, out)
	}

	s := string(bytes.TrimSpace(out))
	if s != "false" {
		t.Errorf("in non-test testing.Test() returned %q, want %q", s, "false")
	}
}

// runTest runs a helper test with -test.v, ignoring its exit status.
// runTest both logs and returns the test output.
func runTest(t *testing.T, test string) []byte {
	t.Helper()

	testenv.MustHaveExec(t)

	exe, err := os.Executable()
	if err != nil {
		t.Skipf("can't find test executable: %v", err)
	}

	cmd := testenv.Command(t, exe, "-test.run=^"+test+"$", "-test.bench="+test, "-test.v", "-test.parallel=2", "-test.benchtime=2x")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	out, err := cmd.CombinedOutput()
	t.Logf("%v: %v\n%s", cmd, err, out)

	return out
}

// doRace provokes a data race that generates a race detector report if run
// under the race detector and is otherwise benign.
func doRace() {
	var x int
	c1 := make(chan bool)
	go func() {
		x = 1 // racy write
		c1 <- true
	}()
	_ = x // racy read
	<-c1
}

func TestRaceReports(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		// Generate a race detector report in a sub test.
		t.Run("Sub", func(t *testing.T) {
			doRace()
		})
		return
	}

	out := runTest(t, "TestRaceReports")

	// We should see at most one race detector report.
	c := bytes.Count(out, []byte("race detected"))
	want := 0
	if race.Enabled {
		want = 1
	}
	if c != want {
		t.Errorf("got %d race reports, want %d", c, want)
	}
}

// Issue #60083. This used to fail on the race builder.
func TestRaceName(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		doRace()
		return
	}

	out := runTest(t, "TestRaceName")

	if regexp.MustCompile(`=== NAME\s*$`).Match(out) {
		t.Errorf("incorrectly reported test with no name")
	}
}

func TestRaceSubReports(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		t.Parallel()
		c1 := make(chan bool, 1)
		t.Run("sub", func(t *testing.T) {
			t.Run("subsub1", func(t *testing.T) {
				t.Parallel()
				doRace()
				c1 <- true
			})
			t.Run("subsub2", func(t *testing.T) {
				t.Parallel()
				doRace()
				<-c1
			})
		})
		doRace()
		return
	}

	out := runTest(t, "TestRaceSubReports")

	// There should be three race reports: one for each subtest, and one for the
	// race after the subtests complete. Note that because the subtests run in
	// parallel, the race stacks may both be printed in with one or the other
	// test's logs.
	cReport := bytes.Count(out, []byte("race detected during execution of test"))
	wantReport := 0
	if race.Enabled {
		wantReport = 3
	}
	if cReport != wantReport {
		t.Errorf("got %d race reports, want %d", cReport, wantReport)
	}

	// Regardless of when the stacks are printed, we expect each subtest to be
	// marked as failed, and that failure should propagate up to the parents.
	cFail := bytes.Count(out, []byte("--- FAIL:"))
	wantFail := 0
	if race.Enabled {
		wantFail = 4
	}
	if cFail != wantFail {
		t.Errorf(`got %d "--- FAIL:" lines, want %d`, cReport, wantReport)
	}
}

func TestRaceInCleanup(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		t.Cleanup(doRace)
		t.Parallel()
		t.Run("sub", func(t *testing.T) {
			t.Parallel()
			// No race should be reported for sub.
		})
		return
	}

	out := runTest(t, "TestRaceInCleanup")

	// There should be one race report, for the parent test only.
	cReport := bytes.Count(out, []byte("race detected during execution of test"))
	wantReport := 0
	if race.Enabled {
		wantReport = 1
	}
	if cReport != wantReport {
		t.Errorf("got %d race reports, want %d", cReport, wantReport)
	}

	// Only the parent test should be marked as failed.
	// (The subtest does not race, and should pass.)
	cFail := bytes.Count(out, []byte("--- FAIL:"))
	wantFail := 0
	if race.Enabled {
		wantFail = 1
	}
	if cFail != wantFail {
		t.Errorf(`got %d "--- FAIL:" lines, want %d`, cReport, wantReport)
	}
}

func TestDeepSubtestRace(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		t.Run("sub", func(t *testing.T) {
			t.Run("subsub", func(t *testing.T) {
				t.Run("subsubsub", func(t *testing.T) {
					doRace()
				})
			})
			doRace()
		})
		return
	}

	out := runTest(t, "TestDeepSubtestRace")

	c := bytes.Count(out, []byte("race detected during execution of test"))
	want := 0
	// There should be two race reports.
	if race.Enabled {
		want = 2
	}
	if c != want {
		t.Errorf("got %d race reports, want %d", c, want)
	}
}

func TestRaceDuringParallelFailsAllSubtests(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		var ready sync.WaitGroup
		ready.Add(2)
		done := make(chan struct{})
		go func() {
			ready.Wait()
			doRace() // This race happens while both subtests are running.
			close(done)
		}()

		t.Run("sub", func(t *testing.T) {
			t.Run("subsub1", func(t *testing.T) {
				t.Parallel()
				ready.Done()
				<-done
			})
			t.Run("subsub2", func(t *testing.T) {
				t.Parallel()
				ready.Done()
				<-done
			})
		})

		return
	}

	out := runTest(t, "TestRaceDuringParallelFailsAllSubtests")

	c := bytes.Count(out, []byte("race detected during execution of test"))
	want := 0
	// Each subtest should report the race independently.
	if race.Enabled {
		want = 2
	}
	if c != want {
		t.Errorf("got %d race reports, want %d", c, want)
	}
}

func TestRaceBeforeParallel(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		t.Run("sub", func(t *testing.T) {
			doRace()
			t.Parallel()
		})
		return
	}

	out := runTest(t, "TestRaceBeforeParallel")

	c := bytes.Count(out, []byte("race detected during execution of test"))
	want := 0
	// We should see one race detector report.
	if race.Enabled {
		want = 1
	}
	if c != want {
		t.Errorf("got %d race reports, want %d", c, want)
	}
}

func TestRaceBeforeTests(t *testing.T) {
	testenv.MustHaveExec(t)

	exe, err := os.Executable()
	if err != nil {
		t.Skipf("can't find test executable: %v", err)
	}

	cmd := testenv.Command(t, exe, "-test.run=^$")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_RACE_BEFORE_TESTS=1")
	out, _ := cmd.CombinedOutput()
	t.Logf("%s", out)

	c := bytes.Count(out, []byte("race detected outside of test execution"))

	want := 0
	if race.Enabled {
		want = 1
	}
	if c != want {
		t.Errorf("got %d race reports; want %d", c, want)
	}
}

func TestBenchmarkRace(t *testing.T) {
	out := runTest(t, "BenchmarkRacy")
	c := bytes.Count(out, []byte("race detected during execution of test"))

	want := 0
	// We should see one race detector report.
	if race.Enabled {
		want = 1
	}
	if c != want {
		t.Errorf("got %d race reports; want %d", c, want)
	}
}

func BenchmarkRacy(b *testing.B) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		b.Skipf("skipping intentionally-racy benchmark")
	}
	for i := 0; i < b.N; i++ {
		doRace()
	}
}

func TestBenchmarkSubRace(t *testing.T) {
	out := runTest(t, "BenchmarkSubRacy")
	c := bytes.Count(out, []byte("race detected during execution of test"))

	want := 0
	// We should see two race detector reports:
	// one in the sub-bencmark, and one in the parent afterward.
	if race.Enabled {
		want = 2
	}
	if c != want {
		t.Errorf("got %d race reports; want %d", c, want)
	}
}

func BenchmarkSubRacy(b *testing.B) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		b.Skipf("skipping intentionally-racy benchmark")
	}

	b.Run("non-racy", func(b *testing.B) {
		tot := 0
		for i := 0; i < b.N; i++ {
			tot++
		}
		_ = tot
	})

	b.Run("racy", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			doRace()
		}
	})

	doRace() // should be reported separately
}

func TestRunningTests(t *testing.T) {
	t.Parallel()

	// Regression test for https://go.dev/issue/64404:
	// on timeout, the "running tests" message should not include
	// tests that are waiting on parked subtests.

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		for i := 0; i < 2; i++ {
			t.Run(fmt.Sprintf("outer%d", i), func(t *testing.T) {
				t.Parallel()
				for j := 0; j < 2; j++ {
					t.Run(fmt.Sprintf("inner%d", j), func(t *testing.T) {
						t.Parallel()
						for {
							time.Sleep(1 * time.Millisecond)
						}
					})
				}
			})
		}
	}

	timeout := 10 * time.Millisecond
	for {
		cmd := testenv.Command(t, os.Args[0], "-test.run=^"+t.Name()+"$", "-test.timeout="+timeout.String(), "-test.parallel=4")
		cmd.Env = append(cmd.Environ(), "GO_WANT_HELPER_PROCESS=1")
		out, err := cmd.CombinedOutput()
		t.Logf("%v:\n%s", cmd, out)
		if _, ok := err.(*exec.ExitError); !ok {
			t.Fatal(err)
		}

		// Because the outer subtests (and TestRunningTests itself) are marked as
		// parallel, their test functions return (and are no longer “running”)
		// before the inner subtests are released to run and hang.
		// Only those inner subtests should be reported as running.
		want := []string{
			"TestRunningTests/outer0/inner0",
			"TestRunningTests/outer0/inner1",
			"TestRunningTests/outer1/inner0",
			"TestRunningTests/outer1/inner1",
		}

		got, ok := parseRunningTests(out)
		if slices.Equal(got, want) {
			break
		}
		if ok {
			t.Logf("found running tests:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		} else {
			t.Logf("no running tests found")
		}
		t.Logf("retrying with longer timeout")
		timeout *= 2
	}
}

func TestRunningTestsInCleanup(t *testing.T) {
	t.Parallel()

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		for i := 0; i < 2; i++ {
			t.Run(fmt.Sprintf("outer%d", i), func(t *testing.T) {
				// Not parallel: we expect to see only one outer test,
				// stuck in cleanup after its subtest finishes.

				t.Cleanup(func() {
					for {
						time.Sleep(1 * time.Millisecond)
					}
				})

				for j := 0; j < 2; j++ {
					t.Run(fmt.Sprintf("inner%d", j), func(t *testing.T) {
						t.Parallel()
					})
				}
			})
		}
	}

	timeout := 10 * time.Millisecond
	for {
		cmd := testenv.Command(t, os.Args[0], "-test.run=^"+t.Name()+"$", "-test.timeout="+timeout.String())
		cmd.Env = append(cmd.Environ(), "GO_WANT_HELPER_PROCESS=1")
		out, err := cmd.CombinedOutput()
		t.Logf("%v:\n%s", cmd, out)
		if _, ok := err.(*exec.ExitError); !ok {
			t.Fatal(err)
		}

		// TestRunningTestsInCleanup is blocked in the call to t.Run,
		// but its test function has not yet returned so it should still
		// be considered to be running.
		// outer1 hasn't even started yet, so only outer0 and the top-level
		// test function should be reported as running.
		want := []string{
			"TestRunningTestsInCleanup",
			"TestRunningTestsInCleanup/outer0",
		}

		got, ok := parseRunningTests(out)
		if slices.Equal(got, want) {
			break
		}
		if ok {
			t.Logf("found running tests:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		} else {
			t.Logf("no running tests found")
		}
		t.Logf("retrying with longer timeout")
		timeout *= 2
	}
}

func parseRunningTests(out []byte) (runningTests []string, ok bool) {
	inRunningTests := false
	for _, line := range strings.Split(string(out), "\n") {
		if inRunningTests {
			// Package testing adds one tab, the panic printer adds another.
			if trimmed, ok := strings.CutPrefix(line, "\t\t"); ok {
				if name, _, ok := strings.Cut(trimmed, " "); ok {
					runningTests = append(runningTests, name)
					continue
				}
			}

			// This line is not the name of a running test.
			return runningTests, true
		}

		if strings.TrimSpace(line) == "running tests:" {
			inRunningTests = true
		}
	}

	return nil, false
}

func TestConcurrentRun(t *testing.T) {
	// Regression test for https://go.dev/issue/64402:
	// this deadlocked after https://go.dev/cl/506755.

	block := make(chan struct{})
	var ready, done sync.WaitGroup
	for i := 0; i < 2; i++ {
		ready.Add(1)
		done.Add(1)
		go t.Run("", func(*testing.T) {
			ready.Done()
			<-block
			done.Done()
		})
	}
	ready.Wait()
	close(block)
	done.Wait()
}

func TestParentRun(t1 *testing.T) {
	// Regression test for https://go.dev/issue/64402:
	// this deadlocked after https://go.dev/cl/506755.

	t1.Run("outer", func(t2 *testing.T) {
		t2.Log("Hello outer!")
		t1.Run("not_inner", func(t3 *testing.T) { // Note: this is t1.Run, not t2.Run.
			t3.Log("Hello inner!")
		})
	})
}

func TestContext(t *testing.T) {
	ctx := t.Context()
	if err := ctx.Err(); err != nil {
		t.Fatalf("expected non-canceled context, got %v", err)
	}

	var innerCtx context.Context
	t.Run("inner", func(t *testing.T) {
		innerCtx = t.Context()
		if err := innerCtx.Err(); err != nil {
			t.Fatalf("expected inner test to not inherit canceled context, got %v", err)
		}
	})
	t.Run("inner2", func(t *testing.T) {
		if !errors.Is(innerCtx.Err(), context.Canceled) {
			t.Fatal("expected context of sibling test to be canceled after its test function finished")
		}
	})

	t.Cleanup(func() {
		if !errors.Is(ctx.Err(), context.Canceled) {
			t.Fatal("expected context canceled before cleanup")
		}
	})
}
