// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris
// +build cgo

package signal_test

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal/internal/pty"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestTerminalSignal(t *testing.T) {
	if os.Getenv("GO_TEST_TERMINAL_SIGNALS") != "" {
		var b [1]byte
		fmt.Println("entering read")
		n, err := os.Stdin.Read(b[:])
		fmt.Printf("read %d bytes: %q\n", n, b)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		os.Exit(0)
	}

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

	// The test only fails when using a "slow device," in this
	// case a pseudo-terminal.

	master, sname, err := pty.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer master.Close()
	slave, err := os.OpenFile(sname, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer slave.Close()

	// Start an interactive shell.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bash, "-i")
	cmd.Stdin = slave
	cmd.Stdout = slave
	cmd.Stderr = slave
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setsid:  true,
		Setctty: true,
		Ctty:    int(slave.Fd()),
	}

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	if err := slave.Close(); err != nil {
		t.Errorf("closing slave: %v", err)
	}

	// Read data from master in the background.
	go func() {
		buf := bufio.NewReader(master)
		for {
			data, err := buf.ReadBytes('\n')
			if len(data) > 0 {
				t.Logf("%q", data)
			}
			if err != nil {
				if perr, ok := err.(*os.PathError); ok {
					err = perr.Err
				}
				// EOF means master is closed.
				// EIO means child process is done.
				// "file already closed" means deferred close of master has happened.
				if err != io.EOF && err != syscall.EIO && !strings.Contains(err.Error(), "file already closed") {
					t.Logf("error reading from master: %v", err)
				}
				return
			}
		}
	}()

	// Start a small program that reads from stdin
	// (namely the code at the top of this function).
	if _, err := master.Write([]byte("GO_TEST_TERMINAL_SIGNALS=1 " + os.Args[0] + " -test.run=TestTerminalSignal\n")); err != nil {
		t.Fatal(err)
	}

	// Give the program time to enter the read call.
	time.Sleep(time.Duration(scale) * 100 * time.Millisecond)

	// Send a ^Z to stop the program.
	if _, err := master.Write([]byte{26}); err != nil {
		t.Fatalf("writing ^Z to pty: %v", err)
	}

	// Give the process time to handle the signal.
	time.Sleep(time.Duration(scale) * 100 * time.Millisecond)

	// Restart the stopped program.
	if _, err := master.Write([]byte("fg\n")); err != nil {
		t.Fatalf("writing %q to pty: %v", "fg", err)
	}

	// Write some data for the program to read,
	// which should cause it to exit.
	if _, err := master.Write([]byte{'\n'}); err != nil {
		t.Fatalf("writing %q to pty: %v", "\n", err)
	}

	// Exit the shell with the program's exit status.
	if _, err := master.Write([]byte("exit $?\n")); err != nil {
		t.Fatalf("writing %q to pty: %v", "exit", err)
	}

	if err = cmd.Wait(); err != nil {
		t.Errorf("subprogram failed: %v", err)
	}
}
