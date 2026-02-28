// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test broken pipes on Unix systems.
//
//go:build !plan9 && !js && !wasip1

package os_test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestEPIPE(t *testing.T) {
	// This test cannot be run in parallel because of a race similar
	// to the one reported in https://go.dev/issue/22315.
	//
	// Even though the pipe is opened with O_CLOEXEC, if another test forks in
	// between the call to os.Pipe and the call to r.Close, that child process can
	// retain an open copy of r's file descriptor until it execs. If one of our
	// Write calls occurs during that interval it can spuriously succeed,
	// buffering the write to the child's copy of the pipe (even though the child
	// will not actually read the buffered bytes).

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}

	expect := syscall.EPIPE
	if runtime.GOOS == "windows" {
		// 232 is Windows error code ERROR_NO_DATA, "The pipe is being closed".
		expect = syscall.Errno(232)
	}
	// Every time we write to the pipe we should get an EPIPE.
	for i := 0; i < 20; i++ {
		_, err = w.Write([]byte("hi"))
		if err == nil {
			t.Fatal("unexpected success of Write to broken pipe")
		}
		if pe, ok := err.(*fs.PathError); ok {
			err = pe.Err
		}
		if se, ok := err.(*os.SyscallError); ok {
			err = se.Err
		}
		if err != expect {
			t.Errorf("iteration %d: got %v, expected %v", i, err, expect)
		}
	}
}

func TestStdPipe(t *testing.T) {
	switch runtime.GOOS {
	case "windows":
		t.Skip("Windows doesn't support SIGPIPE")
	}

	if os.Getenv("GO_TEST_STD_PIPE_HELPER") != "" {
		if os.Getenv("GO_TEST_STD_PIPE_HELPER_SIGNAL") != "" {
			signal.Notify(make(chan os.Signal, 1), syscall.SIGPIPE)
		}
		switch os.Getenv("GO_TEST_STD_PIPE_HELPER") {
		case "1":
			os.Stdout.Write([]byte("stdout"))
		case "2":
			os.Stderr.Write([]byte("stderr"))
		case "3":
			if _, err := os.NewFile(3, "3").Write([]byte("3")); err == nil {
				os.Exit(3)
			}
		default:
			panic("unrecognized value for GO_TEST_STD_PIPE_HELPER")
		}
		// For stdout/stderr, we should have crashed with a broken pipe error.
		// The caller will be looking for that exit status,
		// so just exit normally here to cause a failure in the caller.
		// For descriptor 3, a normal exit is expected.
		os.Exit(0)
	}

	testenv.MustHaveExec(t)
	// This test cannot be run in parallel due to the same race as for TestEPIPE.
	// (We expect a write to a closed pipe can fail, but a concurrent fork of a
	// child process can cause the pipe to unexpectedly remain open.)

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	// Invoke the test program to run the test and write to a closed pipe.
	// If sig is false:
	// writing to stdout or stderr should cause an immediate SIGPIPE;
	// writing to descriptor 3 should fail with EPIPE and then exit 0.
	// If sig is true:
	// all writes should fail with EPIPE and then exit 0.
	for _, sig := range []bool{false, true} {
		for dest := 1; dest < 4; dest++ {
			cmd := testenv.Command(t, testenv.Executable(t), "-test.run", "TestStdPipe")
			cmd.Stdout = w
			cmd.Stderr = w
			cmd.ExtraFiles = []*os.File{w}
			cmd.Env = append(os.Environ(), fmt.Sprintf("GO_TEST_STD_PIPE_HELPER=%d", dest))
			if sig {
				cmd.Env = append(cmd.Env, "GO_TEST_STD_PIPE_HELPER_SIGNAL=1")
			}
			if err := cmd.Run(); err == nil {
				if !sig && dest < 3 {
					t.Errorf("unexpected success of write to closed pipe %d sig %t in child", dest, sig)
				}
			} else if ee, ok := err.(*exec.ExitError); !ok {
				t.Errorf("unexpected exec error type %T: %v", err, err)
			} else if ws, ok := ee.Sys().(syscall.WaitStatus); !ok {
				t.Errorf("unexpected wait status type %T: %v", ee.Sys(), ee.Sys())
			} else if ws.Signaled() && ws.Signal() == syscall.SIGPIPE {
				if sig || dest > 2 {
					t.Errorf("unexpected SIGPIPE signal for descriptor %d sig %t", dest, sig)
				}
			} else {
				t.Errorf("unexpected exit status %v for descriptor %d sig %t", err, dest, sig)
			}
		}
	}

	// Test redirecting stdout but not stderr.  Issue 40076.
	cmd := testenv.Command(t, testenv.Executable(t), "-test.run", "TestStdPipe")
	cmd.Stdout = w
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	cmd.Env = append(cmd.Environ(), "GO_TEST_STD_PIPE_HELPER=1")
	if err := cmd.Run(); err == nil {
		t.Errorf("unexpected success of write to closed stdout")
	} else if ee, ok := err.(*exec.ExitError); !ok {
		t.Errorf("unexpected exec error type %T: %v", err, err)
	} else if ws, ok := ee.Sys().(syscall.WaitStatus); !ok {
		t.Errorf("unexpected wait status type %T: %v", ee.Sys(), ee.Sys())
	} else if !ws.Signaled() || ws.Signal() != syscall.SIGPIPE {
		t.Errorf("unexpected exit status %v for write to closed stdout", err)
	}
	if output := stderr.Bytes(); len(output) > 0 {
		t.Errorf("unexpected output on stderr: %s", output)
	}
}

func testClosedPipeRace(t *testing.T, read bool) {
	// This test cannot be run in parallel due to the same race as for TestEPIPE.
	// (We expect a write to a closed pipe can fail, but a concurrent fork of a
	// child process can cause the pipe to unexpectedly remain open.)

	limit := 1
	if !read {
		// Get the amount we have to write to overload a pipe
		// with no reader.
		limit = 131073
		if b, err := os.ReadFile("/proc/sys/fs/pipe-max-size"); err == nil {
			if i, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil {
				limit = i + 1
			}
		}
		t.Logf("using pipe write limit of %d", limit)
	}

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	// Close the read end of the pipe in a goroutine while we are
	// writing to the write end, or vice-versa.
	go func() {
		// Give the main goroutine a chance to enter the Read or
		// Write call. This is sloppy but the test will pass even
		// if we close before the read/write.
		time.Sleep(20 * time.Millisecond)

		var err error
		if read {
			err = r.Close()
		} else {
			err = w.Close()
		}
		if err != nil {
			t.Error(err)
		}
	}()

	b := make([]byte, limit)
	if read {
		_, err = r.Read(b[:])
	} else {
		_, err = w.Write(b[:])
	}
	if err == nil {
		t.Error("I/O on closed pipe unexpectedly succeeded")
	} else if pe, ok := err.(*fs.PathError); !ok {
		t.Errorf("I/O on closed pipe returned unexpected error type %T; expected fs.PathError", pe)
	} else if pe.Err != fs.ErrClosed {
		t.Errorf("got error %q but expected %q", pe.Err, fs.ErrClosed)
	} else {
		t.Logf("I/O returned expected error %q", err)
	}
}

func TestClosedPipeRaceRead(t *testing.T) {
	testClosedPipeRace(t, true)
}

func TestClosedPipeRaceWrite(t *testing.T) {
	testClosedPipeRace(t, false)
}

// Issue 20915: Reading on nonblocking fd should not return "waiting
// for unsupported file type." Currently it returns EAGAIN; it is
// possible that in the future it will simply wait for data.
func TestReadNonblockingFd(t *testing.T) {
	switch runtime.GOOS {
	case "windows":
		t.Skip("Windows doesn't support SetNonblock")
	}
	if os.Getenv("GO_WANT_READ_NONBLOCKING_FD") == "1" {
		fd := syscallDescriptor(os.Stdin.Fd())
		syscall.SetNonblock(fd, true)
		defer syscall.SetNonblock(fd, false)
		_, err := os.Stdin.Read(make([]byte, 1))
		if err != nil {
			if perr, ok := err.(*fs.PathError); !ok || perr.Err != syscall.EAGAIN {
				t.Fatalf("read on nonblocking stdin got %q, should have gotten EAGAIN", err)
			}
		}
		os.Exit(0)
	}

	testenv.MustHaveExec(t)
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()
	cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^"+t.Name()+"$")
	cmd.Env = append(cmd.Environ(), "GO_WANT_READ_NONBLOCKING_FD=1")
	cmd.Stdin = r
	output, err := cmd.CombinedOutput()
	t.Logf("%s", output)
	if err != nil {
		t.Errorf("child process failed: %v", err)
	}
}

func TestCloseWithBlockingReadByNewFile(t *testing.T) {
	t.Parallel()

	var p [2]syscallDescriptor
	err := syscall.Pipe(p[:])
	if err != nil {
		t.Fatal(err)
	}
	// os.NewFile returns a blocking mode file.
	testCloseWithBlockingRead(t, os.NewFile(uintptr(p[0]), "reader"), os.NewFile(uintptr(p[1]), "writer"))
}

func TestCloseWithBlockingReadByFd(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	// Calling Fd will put the file into blocking mode.
	_ = r.Fd()
	testCloseWithBlockingRead(t, r, w)
}

// Test that we don't let a blocking read prevent a close.
func testCloseWithBlockingRead(t *testing.T, r, w *os.File) {
	var (
		enteringRead = make(chan struct{})
		done         = make(chan struct{})
	)
	go func() {
		var b [1]byte
		close(enteringRead)
		_, err := r.Read(b[:])
		if err == nil {
			t.Error("I/O on closed pipe unexpectedly succeeded")
		}

		if pe, ok := err.(*fs.PathError); ok {
			err = pe.Err
		}
		if err != io.EOF && err != fs.ErrClosed {
			t.Errorf("got %v, expected EOF or closed", err)
		}
		close(done)
	}()

	// Give the goroutine a chance to enter the Read
	// or Write call. This is sloppy but the test will
	// pass even if we close before the read/write.
	<-enteringRead
	time.Sleep(20 * time.Millisecond)

	if err := r.Close(); err != nil {
		t.Error(err)
	}
	// r.Close has completed, but since we assume r is in blocking mode that
	// probably didn't unblock the call to r.Read. Close w to unblock it.
	w.Close()
	<-done
}

func TestPipeEOF(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}

	testPipeEOF(t, r, w)
}

// testPipeEOF tests that when the write side of a pipe or FIFO is closed,
// a blocked Read call on the reader side returns io.EOF.
//
// This scenario previously failed to unblock the Read call on darwin.
// (See https://go.dev/issue/24164.)
func testPipeEOF(t *testing.T, r io.ReadCloser, w io.WriteCloser) {
	// parkDelay is an arbitrary delay we wait for a pipe-reader goroutine to park
	// before issuing the corresponding write. The test should pass no matter what
	// delay we use, but with a longer delay is has a higher chance of detecting
	// poller bugs.
	parkDelay := 10 * time.Millisecond
	if testing.Short() {
		parkDelay = 100 * time.Microsecond
	}
	writerDone := make(chan struct{})
	defer func() {
		if err := r.Close(); err != nil {
			t.Errorf("error closing reader: %v", err)
		}
		<-writerDone
	}()

	write := make(chan int, 1)
	go func() {
		defer close(writerDone)

		for i := range write {
			time.Sleep(parkDelay)
			_, err := fmt.Fprintf(w, "line %d\n", i)
			if err != nil {
				t.Errorf("error writing to fifo: %v", err)
				return
			}
		}

		time.Sleep(parkDelay)
		if err := w.Close(); err != nil {
			t.Errorf("error closing writer: %v", err)
		}
	}()

	rbuf := bufio.NewReader(r)
	for i := 0; i < 3; i++ {
		write <- i
		b, err := rbuf.ReadBytes('\n')
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s\n", bytes.TrimSpace(b))
	}

	close(write)
	b, err := rbuf.ReadBytes('\n')
	if err != io.EOF || len(b) != 0 {
		t.Errorf(`ReadBytes: %q, %v; want "", io.EOF`, b, err)
	}
}

// Issue 24481.
func TestFdRace(t *testing.T) {
	// This test starts 100 simultaneous goroutines, which could bury a more
	// interesting stack if this or some other test happens to panic. It is also
	// nearly instantaneous, so any latency benefit from running it in parallel
	// would be minimal.

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	var wg sync.WaitGroup
	call := func() {
		defer wg.Done()
		w.Fd()
	}

	const tries = 100
	for i := 0; i < tries; i++ {
		wg.Add(1)
		go call()
	}
	wg.Wait()
}

func TestFdReadRace(t *testing.T) {
	t.Parallel()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	const count = 10

	c := make(chan bool, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		var buf [count]byte
		r.SetReadDeadline(time.Now().Add(time.Minute))
		c <- true
		if _, err := r.Read(buf[:]); os.IsTimeout(err) {
			t.Error("read timed out")
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-c
		// Give the other goroutine a chance to enter the Read.
		// It doesn't matter if this occasionally fails, the test
		// will still pass, it just won't test anything.
		time.Sleep(10 * time.Millisecond)
		r.Fd()

		// The bug was that Fd would hang until Read timed out.
		// If the bug is fixed, then writing to w and closing r here
		// will cause the Read to exit before the timeout expires.
		w.Write(make([]byte, count))
		r.Close()
	}()

	wg.Wait()
}
