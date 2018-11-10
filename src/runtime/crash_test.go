// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"flag"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

var toRemove []string

func TestMain(m *testing.M) {
	status := m.Run()
	for _, file := range toRemove {
		os.RemoveAll(file)
	}
	os.Exit(status)
}

func testEnv(cmd *exec.Cmd) *exec.Cmd {
	if cmd.Env != nil {
		panic("environment already set")
	}
	for _, env := range os.Environ() {
		// Exclude GODEBUG from the environment to prevent its output
		// from breaking tests that are trying to parse other command output.
		if strings.HasPrefix(env, "GODEBUG=") {
			continue
		}
		// Exclude GOTRACEBACK for the same reason.
		if strings.HasPrefix(env, "GOTRACEBACK=") {
			continue
		}
		cmd.Env = append(cmd.Env, env)
	}
	return cmd
}

var testprog struct {
	sync.Mutex
	dir    string
	target map[string]buildexe
}

type buildexe struct {
	exe string
	err error
}

func runTestProg(t *testing.T, binary, name string) string {
	testenv.MustHaveGoBuild(t)

	exe, err := buildTestProg(t, binary)
	if err != nil {
		t.Fatal(err)
	}

	cmd := testEnv(exec.Command(exe, name))
	var b bytes.Buffer
	cmd.Stdout = &b
	cmd.Stderr = &b
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting %s %s: %v", binary, name, err)
	}

	// If the process doesn't complete within 1 minute,
	// assume it is hanging and kill it to get a stack trace.
	p := cmd.Process
	done := make(chan bool)
	go func() {
		scale := 1
		// This GOARCH/GOOS test is copied from cmd/dist/test.go.
		// TODO(iant): Have cmd/dist update the environment variable.
		if runtime.GOARCH == "arm" || runtime.GOOS == "windows" {
			scale = 2
		}
		if s := os.Getenv("GO_TEST_TIMEOUT_SCALE"); s != "" {
			if sc, err := strconv.Atoi(s); err == nil {
				scale = sc
			}
		}

		select {
		case <-done:
		case <-time.After(time.Duration(scale) * time.Minute):
			p.Signal(sigquit)
		}
	}()

	if err := cmd.Wait(); err != nil {
		t.Logf("%s %s exit status: %v", binary, name, err)
	}
	close(done)

	return b.String()
}

func buildTestProg(t *testing.T, binary string, flags ...string) (string, error) {
	checkStaleRuntime(t)

	testprog.Lock()
	defer testprog.Unlock()
	if testprog.dir == "" {
		dir, err := ioutil.TempDir("", "go-build")
		if err != nil {
			t.Fatalf("failed to create temp directory: %v", err)
		}
		testprog.dir = dir
		toRemove = append(toRemove, dir)
	}

	if testprog.target == nil {
		testprog.target = make(map[string]buildexe)
	}
	name := binary
	if len(flags) > 0 {
		name += "_" + strings.Join(flags, "_")
	}
	target, ok := testprog.target[name]
	if ok {
		return target.exe, target.err
	}

	exe := filepath.Join(testprog.dir, name+".exe")
	cmd := exec.Command(testenv.GoToolPath(t), append([]string{"build", "-o", exe}, flags...)...)
	cmd.Dir = "testdata/" + binary
	out, err := testEnv(cmd).CombinedOutput()
	if err != nil {
		target.err = fmt.Errorf("building %s %v: %v\n%s", binary, flags, err, out)
		testprog.target[name] = target
		return "", target.err
	}
	target.exe = exe
	testprog.target[name] = target
	return exe, nil
}

var (
	staleRuntimeOnce sync.Once // guards init of staleRuntimeErr
	staleRuntimeErr  error
)

func checkStaleRuntime(t *testing.T) {
	staleRuntimeOnce.Do(func() {
		// 'go run' uses the installed copy of runtime.a, which may be out of date.
		out, err := testEnv(exec.Command(testenv.GoToolPath(t), "list", "-f", "{{.Stale}}", "runtime")).CombinedOutput()
		if err != nil {
			staleRuntimeErr = fmt.Errorf("failed to execute 'go list': %v\n%v", err, string(out))
			return
		}
		if string(out) != "false\n" {
			t.Logf("go list -f {{.Stale}} runtime:\n%s", out)
			out, err := testEnv(exec.Command(testenv.GoToolPath(t), "list", "-f", "{{.StaleReason}}", "runtime")).CombinedOutput()
			if err != nil {
				t.Logf("go list -f {{.StaleReason}} failed: %v", err)
			}
			t.Logf("go list -f {{.StaleReason}} runtime:\n%s", out)
			staleRuntimeErr = fmt.Errorf("Stale runtime.a. Run 'go install runtime'.")
		}
	})
	if staleRuntimeErr != nil {
		t.Fatal(staleRuntimeErr)
	}
}

func testCrashHandler(t *testing.T, cgo bool) {
	type crashTest struct {
		Cgo bool
	}
	var output string
	if cgo {
		output = runTestProg(t, "testprogcgo", "Crash")
	} else {
		output = runTestProg(t, "testprog", "Crash")
	}
	want := "main: recovered done\nnew-thread: recovered done\nsecond-new-thread: recovered done\nmain-again: recovered done\n"
	if output != want {
		t.Fatalf("output:\n%s\n\nwanted:\n%s", output, want)
	}
}

func TestCrashHandler(t *testing.T) {
	testCrashHandler(t, false)
}

func testDeadlock(t *testing.T, name string) {
	output := runTestProg(t, "testprog", name)
	want := "fatal error: all goroutines are asleep - deadlock!\n"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestSimpleDeadlock(t *testing.T) {
	testDeadlock(t, "SimpleDeadlock")
}

func TestInitDeadlock(t *testing.T) {
	testDeadlock(t, "InitDeadlock")
}

func TestLockedDeadlock(t *testing.T) {
	testDeadlock(t, "LockedDeadlock")
}

func TestLockedDeadlock2(t *testing.T) {
	testDeadlock(t, "LockedDeadlock2")
}

func TestGoexitDeadlock(t *testing.T) {
	output := runTestProg(t, "testprog", "GoexitDeadlock")
	want := "no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
	}
}

func TestStackOverflow(t *testing.T) {
	output := runTestProg(t, "testprog", "StackOverflow")
	want := "runtime: goroutine stack exceeds 1474560-byte limit\nfatal error: stack overflow"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestThreadExhaustion(t *testing.T) {
	output := runTestProg(t, "testprog", "ThreadExhaustion")
	want := "runtime: program exceeds 10-thread limit\nfatal error: thread exhaustion"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestRecursivePanic(t *testing.T) {
	output := runTestProg(t, "testprog", "RecursivePanic")
	want := `wrap: bad
panic: again

`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}

}

func TestGoexitCrash(t *testing.T) {
	output := runTestProg(t, "testprog", "GoexitExit")
	want := "no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
	}
}

func TestGoexitDefer(t *testing.T) {
	c := make(chan struct{})
	go func() {
		defer func() {
			r := recover()
			if r != nil {
				t.Errorf("non-nil recover during Goexit")
			}
			c <- struct{}{}
		}()
		runtime.Goexit()
	}()
	// Note: if the defer fails to run, we will get a deadlock here
	<-c
}

func TestGoNil(t *testing.T) {
	output := runTestProg(t, "testprog", "GoNil")
	want := "go of nil func value"
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
	}
}

func TestMainGoroutineID(t *testing.T) {
	output := runTestProg(t, "testprog", "MainGoroutineID")
	want := "panic: test\n\ngoroutine 1 [running]:\n"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestNoHelperGoroutines(t *testing.T) {
	output := runTestProg(t, "testprog", "NoHelperGoroutines")
	matches := regexp.MustCompile(`goroutine [0-9]+ \[`).FindAllStringSubmatch(output, -1)
	if len(matches) != 1 || matches[0][0] != "goroutine 1 [" {
		t.Fatalf("want to see only goroutine 1, see:\n%s", output)
	}
}

func TestBreakpoint(t *testing.T) {
	output := runTestProg(t, "testprog", "Breakpoint")
	// If runtime.Breakpoint() is inlined, then the stack trace prints
	// "runtime.Breakpoint(...)" instead of "runtime.Breakpoint()".
	want := "runtime.Breakpoint("
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
	}
}

func TestGoexitInPanic(t *testing.T) {
	// see issue 8774: this code used to trigger an infinite recursion
	output := runTestProg(t, "testprog", "GoexitInPanic")
	want := "fatal error: no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

// Issue 14965: Runtime panics should be of type runtime.Error
func TestRuntimePanicWithRuntimeError(t *testing.T) {
	testCases := [...]func(){
		0: func() {
			var m map[uint64]bool
			m[1234] = true
		},
		1: func() {
			ch := make(chan struct{})
			close(ch)
			close(ch)
		},
		2: func() {
			var ch = make(chan struct{})
			close(ch)
			ch <- struct{}{}
		},
		3: func() {
			var s = make([]int, 2)
			_ = s[2]
		},
		4: func() {
			n := -1
			_ = make(chan bool, n)
		},
		5: func() {
			close((chan bool)(nil))
		},
	}

	for i, fn := range testCases {
		got := panicValue(fn)
		if _, ok := got.(runtime.Error); !ok {
			t.Errorf("test #%d: recovered value %v(type %T) does not implement runtime.Error", i, got, got)
		}
	}
}

func panicValue(fn func()) (recovered interface{}) {
	defer func() {
		recovered = recover()
	}()
	fn()
	return
}

func TestPanicAfterGoexit(t *testing.T) {
	// an uncaught panic should still work after goexit
	output := runTestProg(t, "testprog", "PanicAfterGoexit")
	want := "panic: hello"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestRecoveredPanicAfterGoexit(t *testing.T) {
	output := runTestProg(t, "testprog", "RecoveredPanicAfterGoexit")
	want := "fatal error: no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestRecoverBeforePanicAfterGoexit(t *testing.T) {
	// 1. defer a function that recovers
	// 2. defer a function that panics
	// 3. call goexit
	// Goexit should run the #2 defer. Its panic
	// should be caught by the #1 defer, and execution
	// should resume in the caller. Like the Goexit
	// never happened!
	defer func() {
		r := recover()
		if r == nil {
			panic("bad recover")
		}
	}()
	defer func() {
		panic("hello")
	}()
	runtime.Goexit()
}

func TestNetpollDeadlock(t *testing.T) {
	t.Parallel()
	output := runTestProg(t, "testprognet", "NetpollDeadlock")
	want := "done\n"
	if !strings.HasSuffix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestPanicTraceback(t *testing.T) {
	t.Parallel()
	output := runTestProg(t, "testprog", "PanicTraceback")
	want := "panic: hello"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}

	// Check functions in the traceback.
	fns := []string{"main.pt1.func1", "panic", "main.pt2.func1", "panic", "main.pt2", "main.pt1"}
	for _, fn := range fns {
		re := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(fn) + `\(.*\n`)
		idx := re.FindStringIndex(output)
		if idx == nil {
			t.Fatalf("expected %q function in traceback:\n%s", fn, output)
		}
		output = output[idx[1]:]
	}
}

func testPanicDeadlock(t *testing.T, name string, want string) {
	// test issue 14432
	output := runTestProg(t, "testprog", name)
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestPanicDeadlockGosched(t *testing.T) {
	testPanicDeadlock(t, "GoschedInPanic", "panic: errorThatGosched\n\n")
}

func TestPanicDeadlockSyscall(t *testing.T) {
	testPanicDeadlock(t, "SyscallInPanic", "1\n2\npanic: 3\n\n")
}

func TestPanicLoop(t *testing.T) {
	output := runTestProg(t, "testprog", "PanicLoop")
	if want := "panic while printing panic value"; !strings.Contains(output, want) {
		t.Errorf("output does not contain %q:\n%s", want, output)
	}
}

func TestMemPprof(t *testing.T) {
	testenv.MustHaveGoRun(t)

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	got, err := testEnv(exec.Command(exe, "MemProf")).CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	fn := strings.TrimSpace(string(got))
	defer os.Remove(fn)

	for try := 0; try < 2; try++ {
		cmd := testEnv(exec.Command(testenv.GoToolPath(t), "tool", "pprof", "-alloc_space", "-top"))
		// Check that pprof works both with and without explicit executable on command line.
		if try == 0 {
			cmd.Args = append(cmd.Args, exe, fn)
		} else {
			cmd.Args = append(cmd.Args, fn)
		}
		found := false
		for i, e := range cmd.Env {
			if strings.HasPrefix(e, "PPROF_TMPDIR=") {
				cmd.Env[i] = "PPROF_TMPDIR=" + os.TempDir()
				found = true
				break
			}
		}
		if !found {
			cmd.Env = append(cmd.Env, "PPROF_TMPDIR="+os.TempDir())
		}

		top, err := cmd.CombinedOutput()
		t.Logf("%s:\n%s", cmd.Args, top)
		if err != nil {
			t.Error(err)
		} else if !bytes.Contains(top, []byte("MemProf")) {
			t.Error("missing MemProf in pprof output")
		}
	}
}

var concurrentMapTest = flag.Bool("run_concurrent_map_tests", false, "also run flaky concurrent map tests")

func TestConcurrentMapWrites(t *testing.T) {
	if !*concurrentMapTest {
		t.Skip("skipping without -run_concurrent_map_tests")
	}
	testenv.MustHaveGoRun(t)
	output := runTestProg(t, "testprog", "concurrentMapWrites")
	want := "fatal error: concurrent map writes"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}
func TestConcurrentMapReadWrite(t *testing.T) {
	if !*concurrentMapTest {
		t.Skip("skipping without -run_concurrent_map_tests")
	}
	testenv.MustHaveGoRun(t)
	output := runTestProg(t, "testprog", "concurrentMapReadWrite")
	want := "fatal error: concurrent map read and map write"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}
func TestConcurrentMapIterateWrite(t *testing.T) {
	if !*concurrentMapTest {
		t.Skip("skipping without -run_concurrent_map_tests")
	}
	testenv.MustHaveGoRun(t)
	output := runTestProg(t, "testprog", "concurrentMapIterateWrite")
	want := "fatal error: concurrent map iteration and map write"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

type point struct {
	x, y *int
}

func (p *point) negate() {
	*p.x = *p.x * -1
	*p.y = *p.y * -1
}

// Test for issue #10152.
func TestPanicInlined(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("recover failed")
		}
		buf := make([]byte, 2048)
		n := runtime.Stack(buf, false)
		buf = buf[:n]
		if !bytes.Contains(buf, []byte("(*point).negate(")) {
			t.Fatalf("expecting stack trace to contain call to (*point).negate()")
		}
	}()

	pt := new(point)
	pt.negate()
}

// Test for issues #3934 and #20018.
// We want to delay exiting until a panic print is complete.
func TestPanicRace(t *testing.T) {
	testenv.MustHaveGoRun(t)

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	// The test is intentionally racy, and in my testing does not
	// produce the expected output about 0.05% of the time.
	// So run the program in a loop and only fail the test if we
	// get the wrong output ten times in a row.
	const tries = 10
retry:
	for i := 0; i < tries; i++ {
		got, err := testEnv(exec.Command(exe, "PanicRace")).CombinedOutput()
		if err == nil {
			t.Logf("try %d: program exited successfully, should have failed", i+1)
			continue
		}

		if i > 0 {
			t.Logf("try %d:\n", i+1)
		}
		t.Logf("%s\n", got)

		wants := []string{
			"panic: crash",
			"PanicRace",
			"created by ",
		}
		for _, want := range wants {
			if !bytes.Contains(got, []byte(want)) {
				t.Logf("did not find expected string %q", want)
				continue retry
			}
		}

		// Test generated expected output.
		return
	}
	t.Errorf("test ran %d times without producing expected output", tries)
}
