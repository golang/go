// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"internal/asan"
	"internal/msan"
	"internal/profile"
	"internal/race"
	"internal/testenv"
	traceparse "internal/trace"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/trace"
	"strings"
	"sync"
	"testing"
	"time"
)

var toRemove []string

const entrypointVar = "RUNTIME_TEST_ENTRYPOINT"

func TestMain(m *testing.M) {
	switch entrypoint := os.Getenv(entrypointVar); entrypoint {
	case "panic":
		crashViaPanic()
		panic("unreachable")
	case "trap":
		crashViaTrap()
		panic("unreachable")
	default:
		log.Fatalf("invalid %s: %q", entrypointVar, entrypoint)
	case "":
		// fall through to normal behavior
	}

	_, coreErrBefore := os.Stat("core")

	status := m.Run()
	for _, file := range toRemove {
		os.RemoveAll(file)
	}

	_, coreErrAfter := os.Stat("core")
	if coreErrBefore != nil && coreErrAfter == nil {
		fmt.Fprintln(os.Stderr, "runtime.test: some test left a core file behind")
		if status == 0 {
			status = 1
		}
	}

	os.Exit(status)
}

var testprog struct {
	sync.Mutex
	dir    string
	target map[string]*buildexe
}

type buildexe struct {
	once sync.Once
	exe  string
	err  error
}

func runTestProg(t *testing.T, binary, name string, env ...string) string {
	if *flagQuick {
		t.Skip("-quick")
	}

	testenv.MustHaveGoBuild(t)
	t.Helper()

	exe, err := buildTestProg(t, binary)
	if err != nil {
		t.Fatal(err)
	}

	return runBuiltTestProg(t, exe, name, env...)
}

func runBuiltTestProg(t *testing.T, exe, name string, env ...string) string {
	t.Helper()

	if *flagQuick {
		t.Skip("-quick")
	}

	start := time.Now()

	cmd := testenv.CleanCmdEnv(testenv.Command(t, exe, name))
	cmd.Env = append(cmd.Env, env...)
	if testing.Short() {
		cmd.Env = append(cmd.Env, "RUNTIME_TEST_SHORT=1")
	}
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Logf("%v (%v): ok", cmd, time.Since(start))
	} else {
		if _, ok := err.(*exec.ExitError); ok {
			t.Logf("%v: %v", cmd, err)
		} else if errors.Is(err, exec.ErrWaitDelay) {
			t.Fatalf("%v: %v", cmd, err)
		} else {
			t.Fatalf("%v failed to start: %v", cmd, err)
		}
	}
	return string(out)
}

var serializeBuild = make(chan bool, 2)

func buildTestProg(t *testing.T, binary string, flags ...string) (string, error) {
	if *flagQuick {
		t.Skip("-quick")
	}
	testenv.MustHaveGoBuild(t)

	testprog.Lock()
	if testprog.dir == "" {
		dir, err := os.MkdirTemp("", "go-build")
		if err != nil {
			t.Fatalf("failed to create temp directory: %v", err)
		}
		testprog.dir = dir
		toRemove = append(toRemove, dir)
	}

	if testprog.target == nil {
		testprog.target = make(map[string]*buildexe)
	}
	name := binary
	if len(flags) > 0 {
		name += "_" + strings.Join(flags, "_")
	}
	target, ok := testprog.target[name]
	if !ok {
		target = &buildexe{}
		testprog.target[name] = target
	}

	dir := testprog.dir

	// Unlock testprog while actually building, so that other
	// tests can look up executables that were already built.
	testprog.Unlock()

	target.once.Do(func() {
		// Only do two "go build"'s at a time,
		// to keep load from getting too high.
		serializeBuild <- true
		defer func() { <-serializeBuild }()

		// Don't get confused if testenv.GoToolPath calls t.Skip.
		target.err = errors.New("building test called t.Skip")

		if asan.Enabled {
			flags = append(flags, "-asan")
		}
		if msan.Enabled {
			flags = append(flags, "-msan")
		}
		if race.Enabled {
			flags = append(flags, "-race")
		}

		exe := filepath.Join(dir, name+".exe")

		start := time.Now()
		cmd := exec.Command(testenv.GoToolPath(t), append([]string{"build", "-o", exe}, flags...)...)
		t.Logf("running %v", cmd)
		cmd.Dir = "testdata/" + binary
		cmd = testenv.CleanCmdEnv(cmd)

		// Add the rangefunc GOEXPERIMENT unconditionally since some tests depend on it.
		// TODO(61405): Remove this once it's enabled by default.
		edited := false
		for i := range cmd.Env {
			e := cmd.Env[i]
			if _, vars, ok := strings.Cut(e, "GOEXPERIMENT="); ok {
				cmd.Env[i] = "GOEXPERIMENT=" + vars + ",rangefunc"
				edited = true
			}
		}
		if !edited {
			cmd.Env = append(cmd.Env, "GOEXPERIMENT=rangefunc")
		}

		out, err := cmd.CombinedOutput()
		if err != nil {
			target.err = fmt.Errorf("building %s %v: %v\n%s", binary, flags, err, out)
		} else {
			t.Logf("built %v in %v", name, time.Since(start))
			target.exe = exe
			target.err = nil
		}
	})

	return target.exe, target.err
}

func TestVDSO(t *testing.T) {
	t.Parallel()
	output := runTestProg(t, "testprog", "SignalInVDSO")
	want := "success\n"
	if output != want {
		t.Fatalf("output:\n%s\n\nwanted:\n%s", output, want)
	}
}

func testCrashHandler(t *testing.T, cgo bool) {
	type crashTest struct {
		Cgo bool
	}
	var output string
	if cgo {
		if runtime.GOOS == "freebsd" && race.Enabled {
			t.Skipf("race + cgo freebsd not supported. See https://go.dev/issue/73788.")
		}
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

var deadlockBuildTypes = testenv.SpecialBuildTypes{
	// External linking brings in cgo, causing deadlock detection not working.
	Cgo:  false,
	Asan: asan.Enabled,
	Msan: msan.Enabled,
	Race: race.Enabled,
}

func testDeadlock(t *testing.T, name string) {
	// External linking brings in cgo, causing deadlock detection not working.
	testenv.MustInternalLink(t, deadlockBuildTypes)

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
	// External linking brings in cgo, causing deadlock detection not working.
	testenv.MustInternalLink(t, deadlockBuildTypes)

	output := runTestProg(t, "testprog", "GoexitDeadlock")
	want := "no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.Contains(output, want) {
		t.Fatalf("output:\n%s\n\nwant output containing: %s", output, want)
	}
}

func TestStackOverflow(t *testing.T) {
	output := runTestProg(t, "testprog", "StackOverflow")
	want := []string{
		"runtime: goroutine stack exceeds 1474560-byte limit\n",
		"fatal error: stack overflow",
		// information about the current SP and stack bounds
		"runtime: sp=",
		"stack=[",
	}
	if !strings.HasPrefix(output, want[0]) {
		t.Errorf("output does not start with %q", want[0])
	}
	for _, s := range want[1:] {
		if !strings.Contains(output, s) {
			t.Errorf("output does not contain %q", s)
		}
	}
	if t.Failed() {
		t.Logf("output:\n%s", output)
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

func TestRecursivePanic2(t *testing.T) {
	output := runTestProg(t, "testprog", "RecursivePanic2")
	want := `first panic
second panic
panic: third panic

`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}

}

func TestRecursivePanic3(t *testing.T) {
	output := runTestProg(t, "testprog", "RecursivePanic3")
	want := `panic: first panic

`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}

}

func TestRecursivePanic4(t *testing.T) {
	output := runTestProg(t, "testprog", "RecursivePanic4")
	want := `panic: first panic [recovered]
	panic: second panic
`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}

}

func TestRecursivePanic5(t *testing.T) {
	output := runTestProg(t, "testprog", "RecursivePanic5")
	want := `first panic
second panic
panic: third panic
`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}

}

func TestRepanickedPanic(t *testing.T) {
	output := runTestProg(t, "testprog", "RepanickedPanic")
	want := `panic: message [recovered, repanicked]
`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestRepanickedMiddlePanic(t *testing.T) {
	output := runTestProg(t, "testprog", "RepanickedMiddlePanic")
	want := `panic: inner [recovered]
	panic: middle [recovered, repanicked]
	panic: outer
`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestRepanickedPanicSandwich(t *testing.T) {
	output := runTestProg(t, "testprog", "RepanickedPanicSandwich")
	want := `panic: outer [recovered]
	panic: inner [recovered]
	panic: outer
`
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestGoexitCrash(t *testing.T) {
	// External linking brings in cgo, causing deadlock detection not working.
	testenv.MustInternalLink(t, deadlockBuildTypes)

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
	// External linking brings in cgo, causing deadlock detection not working.
	testenv.MustInternalLink(t, deadlockBuildTypes)

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

func panicValue(fn func()) (recovered any) {
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
	// External linking brings in cgo, causing deadlock detection not working.
	testenv.MustInternalLink(t, deadlockBuildTypes)

	output := runTestProg(t, "testprog", "RecoveredPanicAfterGoexit")
	want := "fatal error: no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestRecoverBeforePanicAfterGoexit(t *testing.T) {
	// External linking brings in cgo, causing deadlock detection not working.
	testenv.MustInternalLink(t, deadlockBuildTypes)

	t.Parallel()
	output := runTestProg(t, "testprog", "RecoverBeforePanicAfterGoexit")
	want := "fatal error: no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestRecoverBeforePanicAfterGoexit2(t *testing.T) {
	// External linking brings in cgo, causing deadlock detection not working.
	testenv.MustInternalLink(t, deadlockBuildTypes)

	t.Parallel()
	output := runTestProg(t, "testprog", "RecoverBeforePanicAfterGoexit2")
	want := "fatal error: no goroutines (main called runtime.Goexit) - deadlock!"
	if !strings.HasPrefix(output, want) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
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
	want := "panic: hello\n\tpanic: panic pt2\n\tpanic: panic pt1\n"
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

	got, err := testenv.CleanCmdEnv(exec.Command(exe, "MemProf")).CombinedOutput()
	if err != nil {
		t.Fatalf("testprog failed: %s, output:\n%s", err, got)
	}
	fn := strings.TrimSpace(string(got))
	defer os.Remove(fn)

	for try := 0; try < 2; try++ {
		cmd := testenv.CleanCmdEnv(exec.Command(testenv.GoToolPath(t), "tool", "pprof", "-alloc_space", "-top"))
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
	if race.Enabled {
		t.Skip("skipping test: -race will catch the race, this test is for the built-in race detection")
	}
	testenv.MustHaveGoRun(t)
	output := runTestProg(t, "testprog", "concurrentMapWrites")
	want := "fatal error: concurrent map writes\n"
	// Concurrent writes can corrupt the map in a way that we
	// detect with a separate throw.
	want2 := "fatal error: small map with no empty slot (concurrent map writes?)\n"
	if !strings.HasPrefix(output, want) && !strings.HasPrefix(output, want2) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}
func TestConcurrentMapReadWrite(t *testing.T) {
	if !*concurrentMapTest {
		t.Skip("skipping without -run_concurrent_map_tests")
	}
	if race.Enabled {
		t.Skip("skipping test: -race will catch the race, this test is for the built-in race detection")
	}
	testenv.MustHaveGoRun(t)
	output := runTestProg(t, "testprog", "concurrentMapReadWrite")
	want := "fatal error: concurrent map read and map write\n"
	// Concurrent writes can corrupt the map in a way that we
	// detect with a separate throw.
	want2 := "fatal error: small map with no empty slot (concurrent map writes?)\n"
	if !strings.HasPrefix(output, want) && !strings.HasPrefix(output, want2) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}
func TestConcurrentMapIterateWrite(t *testing.T) {
	if !*concurrentMapTest {
		t.Skip("skipping without -run_concurrent_map_tests")
	}
	if race.Enabled {
		t.Skip("skipping test: -race will catch the race, this test is for the built-in race detection")
	}
	testenv.MustHaveGoRun(t)
	output := runTestProg(t, "testprog", "concurrentMapIterateWrite")
	want := "fatal error: concurrent map iteration and map write\n"
	// Concurrent writes can corrupt the map in a way that we
	// detect with a separate throw.
	want2 := "fatal error: small map with no empty slot (concurrent map writes?)\n"
	if !strings.HasPrefix(output, want) && !strings.HasPrefix(output, want2) {
		t.Fatalf("output does not start with %q:\n%s", want, output)
	}
}

func TestConcurrentMapWritesIssue69447(t *testing.T) {
	testenv.MustHaveGoRun(t)
	if race.Enabled {
		t.Skip("skipping test: -race will catch the race, this test is for the built-in race detection")
	}
	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200; i++ {
		output := runBuiltTestProg(t, exe, "concurrentMapWrites")
		if output == "" {
			// If we didn't detect an error, that's ok.
			// This case makes this test not flaky like
			// the other ones above.
			// (More correctly, this case makes this test flaky
			// in the other direction, in that it might not
			// detect a problem even if there is one.)
			continue
		}
		want := "fatal error: concurrent map writes\n"
		// Concurrent writes can corrupt the map in a way that we
		// detect with a separate throw.
		want2 := "fatal error: small map with no empty slot (concurrent map writes?)\n"
		if !strings.HasPrefix(output, want) && !strings.HasPrefix(output, want2) {
			t.Fatalf("output does not start with %q:\n%s", want, output)
		}
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
		got, err := testenv.CleanCmdEnv(exec.Command(exe, "PanicRace")).CombinedOutput()
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

func TestBadTraceback(t *testing.T) {
	if asan.Enabled || msan.Enabled || race.Enabled {
		t.Skip("skipped test: checkptr mode catches the corruption")
	}
	output := runTestProg(t, "testprog", "BadTraceback")
	for _, want := range []string{
		"unexpected return pc",
		"called from 0xbad",
		"00000bad",    // Smashed LR in hex dump
		"<main.badLR", // Symbolization in hex dump (badLR1 or badLR2)
	} {
		if !strings.Contains(output, want) {
			t.Errorf("output does not contain %q:\n%s", want, output)
		}
	}
}

func TestTimePprof(t *testing.T) {
	// This test is unreliable on any system in which nanotime
	// calls into libc.
	switch runtime.GOOS {
	case "aix", "darwin", "illumos", "openbsd", "solaris":
		t.Skipf("skipping on %s because nanotime calls libc", runtime.GOOS)
	}
	if race.Enabled || asan.Enabled || msan.Enabled {
		t.Skip("skipping on sanitizers because the sanitizer runtime is external code")
	}

	// Pass GOTRACEBACK for issue #41120 to try to get more
	// information on timeout.
	fn := runTestProg(t, "testprog", "TimeProf", "GOTRACEBACK=crash")
	fn = strings.TrimSpace(fn)
	defer os.Remove(fn)

	cmd := testenv.CleanCmdEnv(exec.Command(testenv.GoToolPath(t), "tool", "pprof", "-top", "-nodecount=1", fn))
	cmd.Env = append(cmd.Env, "PPROF_TMPDIR="+os.TempDir())
	top, err := cmd.CombinedOutput()
	t.Logf("%s", top)
	if err != nil {
		t.Error(err)
	} else if bytes.Contains(top, []byte("ExternalCode")) {
		t.Error("profiler refers to ExternalCode")
	}
}

// Test that runtime.abort does so.
func TestAbort(t *testing.T) {
	// Pass GOTRACEBACK to ensure we get runtime frames.
	output := runTestProg(t, "testprog", "Abort", "GOTRACEBACK=system")
	if want := "runtime.abort"; !strings.Contains(output, want) {
		t.Errorf("output does not contain %q:\n%s", want, output)
	}
	if strings.Contains(output, "BAD") {
		t.Errorf("output contains BAD:\n%s", output)
	}
	// Check that it's a signal traceback.
	want := "PC="
	// For systems that use a breakpoint, check specifically for that.
	switch runtime.GOARCH {
	case "386", "amd64":
		switch runtime.GOOS {
		case "plan9":
			want = "sys: breakpoint"
		case "windows":
			want = "Exception 0x80000003"
		default:
			want = "SIGTRAP"
		}
	}
	if !strings.Contains(output, want) {
		t.Errorf("output does not contain %q:\n%s", want, output)
	}
}

// For TestRuntimePanic: test a panic in the runtime package without
// involving the testing harness.
func init() {
	if os.Getenv("GO_TEST_RUNTIME_PANIC") == "1" {
		defer func() {
			if r := recover(); r != nil {
				// We expect to crash, so exit 0
				// to indicate failure.
				os.Exit(0)
			}
		}()
		runtime.PanicForTesting(nil, 1)
		// We expect to crash, so exit 0 to indicate failure.
		os.Exit(0)
	}
	if os.Getenv("GO_TEST_RUNTIME_NPE_READMEMSTATS") == "1" {
		runtime.ReadMemStats(nil)
		os.Exit(0)
	}
	if os.Getenv("GO_TEST_RUNTIME_NPE_FUNCMETHOD") == "1" {
		var f *runtime.Func
		_ = f.Entry()
		os.Exit(0)
	}

}

func TestRuntimePanic(t *testing.T) {
	cmd := testenv.CleanCmdEnv(exec.Command(testenv.Executable(t), "-test.run=^TestRuntimePanic$"))
	cmd.Env = append(cmd.Env, "GO_TEST_RUNTIME_PANIC=1")
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	if err == nil {
		t.Error("child process did not fail")
	} else if want := "runtime.unexportedPanicForTesting"; !bytes.Contains(out, []byte(want)) {
		t.Errorf("output did not contain expected string %q", want)
	}
}

func TestTracebackRuntimeFunction(t *testing.T) {
	cmd := testenv.CleanCmdEnv(exec.Command(testenv.Executable(t), "-test.run=^TestTracebackRuntimeFunction$"))
	cmd.Env = append(cmd.Env, "GO_TEST_RUNTIME_NPE_READMEMSTATS=1")
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	if err == nil {
		t.Error("child process did not fail")
	} else if want := "runtime.ReadMemStats"; !bytes.Contains(out, []byte(want)) {
		t.Errorf("output did not contain expected string %q", want)
	}
}

func TestTracebackRuntimeMethod(t *testing.T) {
	cmd := testenv.CleanCmdEnv(exec.Command(testenv.Executable(t), "-test.run=^TestTracebackRuntimeMethod$"))
	cmd.Env = append(cmd.Env, "GO_TEST_RUNTIME_NPE_FUNCMETHOD=1")
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	if err == nil {
		t.Error("child process did not fail")
	} else if want := "runtime.(*Func).Entry"; !bytes.Contains(out, []byte(want)) {
		t.Errorf("output did not contain expected string %q", want)
	}
}

// Test that g0 stack overflows are handled gracefully.
func TestG0StackOverflow(t *testing.T) {
	if runtime.GOOS == "ios" {
		testenv.SkipFlaky(t, 62671)
	}

	if os.Getenv("TEST_G0_STACK_OVERFLOW") != "1" {
		cmd := testenv.CleanCmdEnv(testenv.Command(t, testenv.Executable(t), "-test.run=^TestG0StackOverflow$", "-test.v"))
		cmd.Env = append(cmd.Env, "TEST_G0_STACK_OVERFLOW=1")
		out, err := cmd.CombinedOutput()
		t.Logf("output:\n%s", out)
		// Don't check err since it's expected to crash.
		if n := strings.Count(string(out), "morestack on g0\n"); n != 1 {
			t.Fatalf("%s\n(exit status %v)", out, err)
		}
		if runtime.CrashStackImplemented {
			// check for a stack trace
			want := "runtime.stackOverflow"
			if n := strings.Count(string(out), want); n < 5 {
				t.Errorf("output does not contain %q at least 5 times:\n%s", want, out)
			}
			return // it's not a signal-style traceback
		}
		// Check that it's a signal-style traceback.
		if runtime.GOOS != "windows" {
			if want := "PC="; !strings.Contains(string(out), want) {
				t.Errorf("output does not contain %q:\n%s", want, out)
			}
		}
		return
	}

	runtime.G0StackOverflow()
}

// For TestCrashWhileTracing: test a panic without involving the testing
// harness, as we rely on stdout only containing trace output.
func init() {
	if os.Getenv("TEST_CRASH_WHILE_TRACING") == "1" {
		trace.Start(os.Stdout)
		trace.Log(context.Background(), "xyzzy-cat", "xyzzy-msg")
		panic("yzzyx")
	}
}

func TestCrashWhileTracing(t *testing.T) {
	testenv.MustHaveExec(t)

	cmd := testenv.CleanCmdEnv(testenv.Command(t, testenv.Executable(t)))
	cmd.Env = append(cmd.Env, "TEST_CRASH_WHILE_TRACING=1")
	stdOut, err := cmd.StdoutPipe()
	var errOut bytes.Buffer
	cmd.Stderr = &errOut

	if err := cmd.Start(); err != nil {
		t.Fatalf("could not start subprocess: %v", err)
	}
	r, err := traceparse.NewReader(stdOut)
	if err != nil {
		t.Fatalf("could not create trace.NewReader: %v", err)
	}
	var seen bool
	nSync := 0
	i := 1
loop:
	for ; ; i++ {
		ev, err := r.ReadEvent()
		if err != nil {
			// We may have a broken tail to the trace -- that's OK.
			// We'll make sure we saw at least one complete generation.
			if err != io.EOF {
				t.Logf("error at event %d: %v", i, err)
			}
			break loop
		}
		switch ev.Kind() {
		case traceparse.EventSync:
			nSync = ev.Sync().N
		case traceparse.EventLog:
			v := ev.Log()
			if v.Category == "xyzzy-cat" && v.Message == "xyzzy-msg" {
				// Should we already stop reading here? More events may come, but
				// we're not guaranteeing a fully unbroken trace until the last
				// byte...
				seen = true
			}
		}
	}
	if err := cmd.Wait(); err == nil {
		t.Error("the process should have panicked")
	}
	if nSync <= 1 {
		t.Errorf("expected at least one full generation to have been emitted before the trace was considered broken")
	}
	if !seen {
		t.Errorf("expected one matching log event matching, but none of the %d received trace events match", i)
	}
	t.Logf("stderr output:\n%s", errOut.String())
	needle := "yzzyx\n"
	if n := strings.Count(errOut.String(), needle); n != 1 {
		t.Fatalf("did not find expected panic message %q\n(exit status %v)", needle, err)
	}
}

// Test that panic message is not clobbered.
// See issue 30150.
func TestDoublePanic(t *testing.T) {
	output := runTestProg(t, "testprog", "DoublePanic", "GODEBUG=clobberfree=1")
	wants := []string{"panic: XXX", "panic: YYY"}
	for _, want := range wants {
		if !strings.Contains(output, want) {
			t.Errorf("output:\n%s\n\nwant output containing: %s", output, want)
		}
	}
}

// Test that panic while panicking discards error message
// See issue 52257
func TestPanicWhilePanicking(t *testing.T) {
	tests := []struct {
		Want string
		Func string
	}{
		{
			"panic while printing panic value: important multi-line\n\terror message",
			"ErrorPanic",
		},
		{
			"panic while printing panic value: important multi-line\n\tstringer message",
			"StringerPanic",
		},
		{
			"panic while printing panic value: type",
			"DoubleErrorPanic",
		},
		{
			"panic while printing panic value: type",
			"DoubleStringerPanic",
		},
		{
			"panic while printing panic value: type",
			"CircularPanic",
		},
		{
			"important multi-line\n\tstring message",
			"StringPanic",
		},
		{
			"nil",
			"NilPanic",
		},
	}
	for _, x := range tests {
		output := runTestProg(t, "testprog", x.Func)
		if !strings.Contains(output, x.Want) {
			t.Errorf("output does not contain %q:\n%s", x.Want, output)
		}
	}
}

func TestPanicOnUnsafeSlice(t *testing.T) {
	output := runTestProg(t, "testprog", "panicOnNilAndEleSizeIsZero")
	// Note: This is normally a panic, but is a throw when checkptr is
	// enabled.
	want := "unsafe.Slice: ptr is nil and len is not zero"
	if !strings.Contains(output, want) {
		t.Errorf("output does not contain %q:\n%s", want, output)
	}
}

func TestNetpollWaiters(t *testing.T) {
	t.Parallel()
	output := runTestProg(t, "testprognet", "NetpollWaiters")
	want := "OK\n"
	if output != want {
		t.Fatalf("output is not %q\n%s", want, output)
	}
}

func TestFinalizerOrCleanupDeadlock(t *testing.T) {
	t.Parallel()

	for _, useCleanup := range []bool{false, true} {
		progName := "Finalizer"
		want := "runtime.runFinalizers"
		if useCleanup {
			progName = "Cleanup"
			want = "runtime.runCleanups"
		}

		// The runtime.runFinalizers/runtime.runCleanups frame should appear in panics, even if
		// runtime frames are normally hidden (GOTRACEBACK=all).
		t.Run("Panic", func(t *testing.T) {
			t.Parallel()
			output := runTestProg(t, "testprog", progName+"Deadlock", "GOTRACEBACK=all", "GO_TEST_FINALIZER_DEADLOCK=panic")
			want := want + "()"
			if !strings.Contains(output, want) {
				t.Errorf("output does not contain %q:\n%s", want, output)
			}
		})

		// The runtime.runFinalizers/runtime.Cleanups frame should appear in runtime.Stack,
		// even though runtime frames are normally hidden.
		t.Run("Stack", func(t *testing.T) {
			t.Parallel()
			output := runTestProg(t, "testprog", progName+"Deadlock", "GO_TEST_FINALIZER_DEADLOCK=stack")
			want := want + "()"
			if !strings.Contains(output, want) {
				t.Errorf("output does not contain %q:\n%s", want, output)
			}
		})

		// The runtime.runFinalizers/runtime.Cleanups frame should appear in goroutine
		// profiles.
		t.Run("PprofProto", func(t *testing.T) {
			t.Parallel()
			output := runTestProg(t, "testprog", progName+"Deadlock", "GO_TEST_FINALIZER_DEADLOCK=pprof_proto")

			p, err := profile.Parse(strings.NewReader(output))
			if err != nil {
				// Logging the binary proto data is not very nice, but it might
				// be a text error message instead.
				t.Logf("Output: %s", output)
				t.Fatalf("Error parsing proto output: %v", err)
			}
			for _, s := range p.Sample {
				for _, loc := range s.Location {
					for _, line := range loc.Line {
						if line.Function.Name == want {
							// Done!
							return
						}
					}
				}
			}
			t.Errorf("Profile does not contain %q:\n%s", want, p)
		})

		// The runtime.runFinalizers/runtime.runCleanups frame should appear in goroutine
		// profiles (debug=1).
		t.Run("PprofDebug1", func(t *testing.T) {
			t.Parallel()
			output := runTestProg(t, "testprog", progName+"Deadlock", "GO_TEST_FINALIZER_DEADLOCK=pprof_debug1")
			want := want + "+"
			if !strings.Contains(output, want) {
				t.Errorf("output does not contain %q:\n%s", want, output)
			}
		})

		// The runtime.runFinalizers/runtime.runCleanups frame should appear in goroutine
		// profiles (debug=2).
		t.Run("PprofDebug2", func(t *testing.T) {
			t.Parallel()
			output := runTestProg(t, "testprog", progName+"Deadlock", "GO_TEST_FINALIZER_DEADLOCK=pprof_debug2")
			want := want + "()"
			if !strings.Contains(output, want) {
				t.Errorf("output does not contain %q:\n%s", want, output)
			}
		})
	}
}
