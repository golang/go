// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package runtime_test

import (
	"bytes"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"sync"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

func init() {
	if runtime.Sigisblocked(int(syscall.SIGQUIT)) {
		// We can't use SIGQUIT to kill subprocesses because
		// it's blocked. Use SIGKILL instead. See issue
		// #19196 for an example of when this happens.
		testenv.Sigquit = syscall.SIGKILL
	}
}

func TestBadOpen(t *testing.T) {
	// make sure we get the correct error code if open fails. Same for
	// read/write/close on the resulting -1 fd. See issue 10052.
	nonfile := []byte("/notreallyafile")
	fd := runtime.Open(&nonfile[0], 0, 0)
	if fd != -1 {
		t.Errorf("open(%q)=%d, want -1", nonfile, fd)
	}
	var buf [32]byte
	r := runtime.Read(-1, unsafe.Pointer(&buf[0]), int32(len(buf)))
	if got, want := r, -int32(syscall.EBADF); got != want {
		t.Errorf("read()=%d, want %d", got, want)
	}
	w := runtime.Write(^uintptr(0), unsafe.Pointer(&buf[0]), int32(len(buf)))
	if got, want := w, -int32(syscall.EBADF); got != want {
		t.Errorf("write()=%d, want %d", got, want)
	}
	c := runtime.Close(-1)
	if c != -1 {
		t.Errorf("close()=%d, want -1", c)
	}
}

func TestCrashDumpsAllThreads(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}

	switch runtime.GOOS {
	case "darwin", "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "illumos", "solaris":
	default:
		t.Skipf("skipping; not supported on %v", runtime.GOOS)
	}

	if runtime.GOOS == "openbsd" && (runtime.GOARCH == "arm" || runtime.GOARCH == "mips64") {
		// This may be ncpu < 2 related...
		t.Skipf("skipping; test fails on %s/%s - see issue #42464", runtime.GOOS, runtime.GOARCH)
	}

	if runtime.Sigisblocked(int(syscall.SIGQUIT)) {
		t.Skip("skipping; SIGQUIT is blocked, see golang.org/issue/19196")
	}

	testenv.MustHaveGoBuild(t)

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(exe, "CrashDumpsAllThreads")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env,
		"GOTRACEBACK=crash",
		// Set GOGC=off. Because of golang.org/issue/10958, the tight
		// loops in the test program are not preemptible. If GC kicks
		// in, it may lock up and prevent main from saying it's ready.
		"GOGC=off",
		// Set GODEBUG=asyncpreemptoff=1. If a thread is preempted
		// when it receives SIGQUIT, it won't show the expected
		// stack trace. See issue 35356.
		"GODEBUG=asyncpreemptoff=1",
	)

	var outbuf bytes.Buffer
	cmd.Stdout = &outbuf
	cmd.Stderr = &outbuf

	rp, wp, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer rp.Close()

	cmd.ExtraFiles = []*os.File{wp}

	if err := cmd.Start(); err != nil {
		wp.Close()
		t.Fatalf("starting program: %v", err)
	}

	if err := wp.Close(); err != nil {
		t.Logf("closing write pipe: %v", err)
	}
	if _, err := rp.Read(make([]byte, 1)); err != nil {
		t.Fatalf("reading from pipe: %v", err)
	}

	if err := cmd.Process.Signal(syscall.SIGQUIT); err != nil {
		t.Fatalf("signal: %v", err)
	}

	// No point in checking the error return from Wait--we expect
	// it to fail.
	cmd.Wait()

	// We want to see a stack trace for each thread.
	// Before https://golang.org/cl/2811 running threads would say
	// "goroutine running on other thread; stack unavailable".
	out := outbuf.Bytes()
	n := bytes.Count(out, []byte("main.crashDumpsAllThreadsLoop("))
	if n != 4 {
		t.Errorf("found %d instances of main.crashDumpsAllThreadsLoop; expected 4", n)
		t.Logf("%s", out)
	}
}

func TestPanicSystemstack(t *testing.T) {
	// Test that GOTRACEBACK=crash prints both the system and user
	// stack of other threads.

	// The GOTRACEBACK=crash handler takes 0.1 seconds even if
	// it's not writing a core file and potentially much longer if
	// it is. Skip in short mode.
	if testing.Short() {
		t.Skip("Skipping in short mode (GOTRACEBACK=crash is slow)")
	}

	if runtime.Sigisblocked(int(syscall.SIGQUIT)) {
		t.Skip("skipping; SIGQUIT is blocked, see golang.org/issue/19196")
	}

	t.Parallel()
	cmd := exec.Command(os.Args[0], "testPanicSystemstackInternal")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GOTRACEBACK=crash")
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal("creating pipe: ", err)
	}
	cmd.Stderr = pw
	if err := cmd.Start(); err != nil {
		t.Fatal("starting command: ", err)
	}
	defer cmd.Process.Wait()
	defer cmd.Process.Kill()
	if err := pw.Close(); err != nil {
		t.Log("closing write pipe: ", err)
	}
	defer pr.Close()

	// Wait for "x\nx\n" to indicate almost-readiness.
	buf := make([]byte, 4)
	_, err = io.ReadFull(pr, buf)
	if err != nil || string(buf) != "x\nx\n" {
		t.Fatal("subprocess failed; output:\n", string(buf))
	}

	// The child blockers print "x\n" and then block on a lock. Receiving
	// those bytes only indicates that the child is _about to block_. Since
	// we don't have a way to know when it is fully blocked, sleep a bit to
	// make us less likely to lose the race and signal before the child
	// blocks.
	time.Sleep(100 * time.Millisecond)

	// Send SIGQUIT.
	if err := cmd.Process.Signal(syscall.SIGQUIT); err != nil {
		t.Fatal("signaling subprocess: ", err)
	}

	// Get traceback.
	tb, err := io.ReadAll(pr)
	if err != nil {
		t.Fatal("reading traceback from pipe: ", err)
	}

	// Traceback should have two testPanicSystemstackInternal's
	// and two blockOnSystemStackInternal's.
	if bytes.Count(tb, []byte("testPanicSystemstackInternal")) != 2 {
		t.Fatal("traceback missing user stack:\n", string(tb))
	} else if bytes.Count(tb, []byte("blockOnSystemStackInternal")) != 2 {
		t.Fatal("traceback missing system stack:\n", string(tb))
	}
}

func init() {
	if len(os.Args) >= 2 && os.Args[1] == "testPanicSystemstackInternal" {
		// Complete any in-flight GCs and disable future ones. We're going to
		// block goroutines on runtime locks, which aren't ever preemptible for the
		// GC to scan them.
		runtime.GC()
		debug.SetGCPercent(-1)
		// Get two threads running on the system stack with
		// something recognizable in the stack trace.
		runtime.GOMAXPROCS(2)
		go testPanicSystemstackInternal()
		testPanicSystemstackInternal()
	}
}

func testPanicSystemstackInternal() {
	runtime.BlockOnSystemStack()
	os.Exit(1) // Should be unreachable.
}

func TestSignalExitStatus(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}
	err = testenv.CleanCmdEnv(exec.Command(exe, "SignalExitStatus")).Run()
	if err == nil {
		t.Error("test program succeeded unexpectedly")
	} else if ee, ok := err.(*exec.ExitError); !ok {
		t.Errorf("error (%v) has type %T; expected exec.ExitError", err, err)
	} else if ws, ok := ee.Sys().(syscall.WaitStatus); !ok {
		t.Errorf("error.Sys (%v) has type %T; expected syscall.WaitStatus", ee.Sys(), ee.Sys())
	} else if !ws.Signaled() || ws.Signal() != syscall.SIGTERM {
		t.Errorf("got %v; expected SIGTERM", ee)
	}
}

func TestSignalIgnoreSIGTRAP(t *testing.T) {
	if runtime.GOOS == "openbsd" {
		testenv.SkipFlaky(t, 49725)
	}

	output := runTestProg(t, "testprognet", "SignalIgnoreSIGTRAP")
	want := "OK\n"
	if output != want {
		t.Fatalf("want %s, got %s\n", want, output)
	}
}

func TestSignalDuringExec(t *testing.T) {
	switch runtime.GOOS {
	case "darwin", "dragonfly", "freebsd", "linux", "netbsd", "openbsd":
	default:
		t.Skipf("skipping test on %s", runtime.GOOS)
	}
	output := runTestProg(t, "testprognet", "SignalDuringExec")
	want := "OK\n"
	if output != want {
		t.Fatalf("want %s, got %s\n", want, output)
	}
}

func TestSignalM(t *testing.T) {
	r, w, errno := runtime.Pipe()
	if errno != 0 {
		t.Fatal(syscall.Errno(errno))
	}
	defer func() {
		runtime.Close(r)
		runtime.Close(w)
	}()
	runtime.Closeonexec(r)
	runtime.Closeonexec(w)

	var want, got int64
	var wg sync.WaitGroup
	ready := make(chan *runtime.M)
	wg.Add(1)
	go func() {
		runtime.LockOSThread()
		want, got = runtime.WaitForSigusr1(r, w, func(mp *runtime.M) {
			ready <- mp
		})
		runtime.UnlockOSThread()
		wg.Done()
	}()
	waitingM := <-ready
	runtime.SendSigusr1(waitingM)

	timer := time.AfterFunc(time.Second, func() {
		// Write 1 to tell WaitForSigusr1 that we timed out.
		bw := byte(1)
		if n := runtime.Write(uintptr(w), unsafe.Pointer(&bw), 1); n != 1 {
			t.Errorf("pipe write failed: %d", n)
		}
	})
	defer timer.Stop()

	wg.Wait()
	if got == -1 {
		t.Fatal("signalM signal not received")
	} else if want != got {
		t.Fatalf("signal sent to M %d, but received on M %d", want, got)
	}
}
