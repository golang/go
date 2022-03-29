// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (darwin || dragonfly || freebsd || (linux && !android) || netbsd || openbsd) && cgo

// Note that this test does not work on Solaris: issue #22849.
// Don't run the test on Android because at least some versions of the
// C library do not define the posix_openpt function.

package signal_test

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	ptypkg "os/signal/internal/pty"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestTerminalSignal(t *testing.T) {
	const enteringRead = "test program entering read"
	if os.Getenv("GO_TEST_TERMINAL_SIGNALS") != "" {
		var b [1]byte
		fmt.Println(enteringRead)
		n, err := os.Stdin.Read(b[:])
		if n == 1 {
			if b[0] == '\n' {
				// This is what we expect
				fmt.Println("read newline")
			} else {
				fmt.Printf("read 1 byte: %q\n", b)
			}
		} else {
			fmt.Printf("read %d bytes\n", n)
		}
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		os.Exit(0)
	}

	t.Parallel()

	// The test requires a shell that uses job control.
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skipf("could not find bash: %v", err)
	}

	scale := 1
	if s := os.Getenv("GO_TEST_TIMEOUT_SCALE"); s != "" {
		if sc, err := strconv.Atoi(s); err == nil {
			scale = sc
		}
	}
	pause := time.Duration(scale) * 10 * time.Millisecond
	wait := time.Duration(scale) * 5 * time.Second

	// The test only fails when using a "slow device," in this
	// case a pseudo-terminal.

	pty, procTTYName, err := ptypkg.Open()
	if err != nil {
		ptyErr := err.(*ptypkg.PtyError)
		if ptyErr.FuncName == "posix_openpt" && ptyErr.Errno == syscall.EACCES {
			t.Skip("posix_openpt failed with EACCES, assuming chroot and skipping")
		}
		t.Fatal(err)
	}
	defer pty.Close()
	procTTY, err := os.OpenFile(procTTYName, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer procTTY.Close()

	// Start an interactive shell.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bash, "--norc", "--noprofile", "-i")
	// Clear HISTFILE so that we don't read or clobber the user's bash history.
	cmd.Env = append(os.Environ(), "HISTFILE=")
	cmd.Stdin = procTTY
	cmd.Stdout = procTTY
	cmd.Stderr = procTTY
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setsid:  true,
		Setctty: true,
		Ctty:    0,
	}

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	if err := procTTY.Close(); err != nil {
		t.Errorf("closing procTTY: %v", err)
	}

	progReady := make(chan bool)
	sawPrompt := make(chan bool, 10)
	const prompt = "prompt> "

	// Read data from pty in the background.
	var wg sync.WaitGroup
	wg.Add(1)
	defer wg.Wait()
	go func() {
		defer wg.Done()
		input := bufio.NewReader(pty)
		var line, handled []byte
		for {
			b, err := input.ReadByte()
			if err != nil {
				if len(line) > 0 || len(handled) > 0 {
					t.Logf("%q", append(handled, line...))
				}
				if perr, ok := err.(*fs.PathError); ok {
					err = perr.Err
				}
				// EOF means pty is closed.
				// EIO means child process is done.
				// "file already closed" means deferred close of pty has happened.
				if err != io.EOF && err != syscall.EIO && !strings.Contains(err.Error(), "file already closed") {
					t.Logf("error reading from pty: %v", err)
				}
				return
			}

			line = append(line, b)

			if b == '\n' {
				t.Logf("%q", append(handled, line...))
				line = nil
				handled = nil
				continue
			}

			if bytes.Contains(line, []byte(enteringRead)) {
				close(progReady)
				handled = append(handled, line...)
				line = nil
			} else if bytes.Contains(line, []byte(prompt)) && !bytes.Contains(line, []byte("PS1=")) {
				sawPrompt <- true
				handled = append(handled, line...)
				line = nil
			}
		}
	}()

	// Set the bash prompt so that we can see it.
	if _, err := pty.Write([]byte("PS1='" + prompt + "'\n")); err != nil {
		t.Fatalf("setting prompt: %v", err)
	}
	select {
	case <-sawPrompt:
	case <-time.After(wait):
		t.Fatal("timed out waiting for shell prompt")
	}

	// Start a small program that reads from stdin
	// (namely the code at the top of this function).
	if _, err := pty.Write([]byte("GO_TEST_TERMINAL_SIGNALS=1 " + os.Args[0] + " -test.run=TestTerminalSignal\n")); err != nil {
		t.Fatal(err)
	}

	// Wait for the program to print that it is starting.
	select {
	case <-progReady:
	case <-time.After(wait):
		t.Fatal("timed out waiting for program to start")
	}

	// Give the program time to enter the read call.
	// It doesn't matter much if we occasionally don't wait long enough;
	// we won't be testing what we want to test, but the overall test
	// will pass.
	time.Sleep(pause)

	// Send a ^Z to stop the program.
	if _, err := pty.Write([]byte{26}); err != nil {
		t.Fatalf("writing ^Z to pty: %v", err)
	}

	// Wait for the program to stop and return to the shell.
	select {
	case <-sawPrompt:
	case <-time.After(wait):
		t.Fatal("timed out waiting for shell prompt")
	}

	// Restart the stopped program.
	if _, err := pty.Write([]byte("fg\n")); err != nil {
		t.Fatalf("writing %q to pty: %v", "fg", err)
	}

	// Give the process time to restart.
	// This is potentially racy: if the process does not restart
	// quickly enough then the byte we send will go to bash rather
	// than the program. Unfortunately there isn't anything we can
	// look for to know that the program is running again.
	// bash will print the program name, but that happens before it
	// restarts the program.
	time.Sleep(10 * pause)

	// Write some data for the program to read,
	// which should cause it to exit.
	if _, err := pty.Write([]byte{'\n'}); err != nil {
		t.Fatalf("writing %q to pty: %v", "\n", err)
	}

	// Wait for the program to exit.
	select {
	case <-sawPrompt:
	case <-time.After(wait):
		t.Fatal("timed out waiting for shell prompt")
	}

	// Exit the shell with the program's exit status.
	if _, err := pty.Write([]byte("exit $?\n")); err != nil {
		t.Fatalf("writing %q to pty: %v", "exit", err)
	}

	if err = cmd.Wait(); err != nil {
		t.Errorf("subprogram failed: %v", err)
	}
}
