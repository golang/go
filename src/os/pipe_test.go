// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test broken pipes on Unix systems.
// +build !windows,!plan9,!nacl

package os_test

import (
	"fmt"
	"internal/testenv"
	"os"
	osexec "os/exec"
	"os/signal"
	"syscall"
	"testing"
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
				t.Errorf("unexpected exit status %v for descriptor %ds sig %t", err, dest, sig)
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
