// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test broken pipes on Unix systems.
// +build !windows,!plan9,!nacl,!js

package os_test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"io/ioutil"
	"os"
	osexec "os/exec"
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
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}

	// Every time we write to the pipe we should get an EPIPE.
	for i := 0; i < 20; i++ {
		_, err = w.Write([]byte("hi"))
		if err == nil {
			t.Fatal("unexpected success of Write to broken pipe")
		}
		if pe, ok := err.(*os.PathError); ok {
			err = pe.Err
		}
		if se, ok := err.(*os.SyscallError); ok {
			err = se.Err
		}
		if err != syscall.EPIPE {
			t.Errorf("iteration %d: got %v, expected EPIPE", i, err)
		}
	}
}

func TestStdPipe(t *testing.T) {
	testenv.MustHaveExec(t)
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
			cmd := osexec.Command(os.Args[0], "-test.run", "TestStdPipeHelper")
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
			} else if ee, ok := err.(*osexec.ExitError); !ok {
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
}

// This is a helper for TestStdPipe. It's not a test in itself.
func TestStdPipeHelper(t *testing.T) {
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
		t.Skip("skipping test helper")
	}
	// For stdout/stderr, we should have crashed with a broken pipe error.
	// The caller will be looking for that exit status,
	// so just exit normally here to cause a failure in the caller.
	// For descriptor 3, a normal exit is expected.
	os.Exit(0)
}

func testClosedPipeRace(t *testing.T, read bool) {
	switch runtime.GOOS {
	case "freebsd":
		t.Skip("FreeBSD does not use the poller; issue 19093")
	}

	limit := 1
	if !read {
		// Get the amount we have to write to overload a pipe
		// with no reader.
		limit = 65537
		if b, err := ioutil.ReadFile("/proc/sys/fs/pipe-max-size"); err == nil {
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
	} else if pe, ok := err.(*os.PathError); !ok {
		t.Errorf("I/O on closed pipe returned unexpected error type %T; expected os.PathError", pe)
	} else if pe.Err != os.ErrClosed {
		t.Errorf("got error %q but expected %q", pe.Err, os.ErrClosed)
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
	if os.Getenv("GO_WANT_READ_NONBLOCKING_FD") == "1" {
		fd := int(os.Stdin.Fd())
		syscall.SetNonblock(fd, true)
		defer syscall.SetNonblock(fd, false)
		_, err := os.Stdin.Read(make([]byte, 1))
		if err != nil {
			if perr, ok := err.(*os.PathError); !ok || perr.Err != syscall.EAGAIN {
				t.Fatalf("read on nonblocking stdin got %q, should have gotten EAGAIN", err)
			}
		}
		os.Exit(0)
	}

	testenv.MustHaveExec(t)
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()
	cmd := osexec.Command(os.Args[0], "-test.run="+t.Name())
	cmd.Env = append(os.Environ(), "GO_WANT_READ_NONBLOCKING_FD=1")
	cmd.Stdin = r
	output, err := cmd.CombinedOutput()
	t.Logf("%s", output)
	if err != nil {
		t.Errorf("child process failed: %v", err)
	}
}

func TestCloseWithBlockingReadByNewFile(t *testing.T) {
	var p [2]int
	err := syscall.Pipe(p[:])
	if err != nil {
		t.Fatal(err)
	}
	// os.NewFile returns a blocking mode file.
	testCloseWithBlockingRead(t, os.NewFile(uintptr(p[0]), "reader"), os.NewFile(uintptr(p[1]), "writer"))
}

func TestCloseWithBlockingReadByFd(t *testing.T) {
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
	defer r.Close()
	defer w.Close()

	c1, c2 := make(chan bool), make(chan bool)
	var wg sync.WaitGroup

	wg.Add(1)
	go func(c chan bool) {
		defer wg.Done()
		// Give the other goroutine a chance to enter the Read
		// or Write call. This is sloppy but the test will
		// pass even if we close before the read/write.
		time.Sleep(20 * time.Millisecond)

		if err := r.Close(); err != nil {
			t.Error(err)
		}
		close(c)
	}(c1)

	wg.Add(1)
	go func(c chan bool) {
		defer wg.Done()
		var b [1]byte
		_, err := r.Read(b[:])
		close(c)
		if err == nil {
			t.Error("I/O on closed pipe unexpectedly succeeded")
		}
		if err != io.EOF {
			t.Errorf("got %v, expected io.EOF", err)
		}
	}(c2)

	for c1 != nil || c2 != nil {
		select {
		case <-c1:
			c1 = nil
			// r.Close has completed, but the blocking Read
			// is hanging. Close the writer to unblock it.
			w.Close()
		case <-c2:
			c2 = nil
		case <-time.After(1 * time.Second):
			switch {
			case c1 != nil && c2 != nil:
				t.Error("timed out waiting for Read and Close")
				w.Close()
			case c1 != nil:
				t.Error("timed out waiting for Close")
			case c2 != nil:
				t.Error("timed out waiting for Read")
			default:
				t.Error("impossible case")
			}
		}
	}

	wg.Wait()
}

// Issue 24164, for pipes.
func TestPipeEOF(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()

		defer func() {
			if err := w.Close(); err != nil {
				t.Errorf("error closing writer: %v", err)
			}
		}()

		for i := 0; i < 3; i++ {
			time.Sleep(10 * time.Millisecond)
			_, err := fmt.Fprintf(w, "line %d\n", i)
			if err != nil {
				t.Errorf("error writing to fifo: %v", err)
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}()

	defer wg.Wait()

	done := make(chan bool)
	go func() {
		defer close(done)

		defer func() {
			if err := r.Close(); err != nil {
				t.Errorf("error closing reader: %v", err)
			}
		}()

		rbuf := bufio.NewReader(r)
		for {
			b, err := rbuf.ReadBytes('\n')
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Error(err)
				return
			}
			t.Logf("%s\n", bytes.TrimSpace(b))
		}
	}()

	select {
	case <-done:
		// Test succeeded.
	case <-time.After(time.Second):
		t.Error("timed out waiting for read")
		// Close the reader to force the read to complete.
		r.Close()
	}
}
