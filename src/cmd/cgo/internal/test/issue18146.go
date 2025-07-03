// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !windows

// Issue 18146: pthread_create failure during syscall.Exec.

package cgotest

import (
	"bytes"
	"crypto/md5"
	"internal/testenv"
	"os"
	"os/exec"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func test18146(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		t.Skipf("skipping flaky test on %s; see golang.org/issue/18202", runtime.GOOS)
	}

	if runtime.GOARCH == "mips" || runtime.GOARCH == "mips64" {
		t.Skipf("skipping on %s", runtime.GOARCH)
	}

	attempts := 1000
	threads := 4

	// Restrict the number of attempts based on RLIMIT_NPROC.
	// Tediously, RLIMIT_NPROC was left out of the syscall package,
	// probably because it is not in POSIX.1, so we define it here.
	// It is not defined on Solaris.
	var nproc int
	setNproc := true
	switch runtime.GOOS {
	default:
		setNproc = false
	case "aix":
		nproc = 9
	case "linux":
		nproc = 6
	case "darwin", "dragonfly", "freebsd", "netbsd", "openbsd":
		nproc = 7
	}
	if setNproc {
		var rlim syscall.Rlimit
		if syscall.Getrlimit(nproc, &rlim) == nil {
			max := int(rlim.Cur) / (threads + 5)
			if attempts > max {
				t.Logf("lowering attempts from %d to %d for RLIMIT_NPROC", attempts, max)
				attempts = max
			}
		}
	}

	if os.Getenv("test18146") == "exec" {
		runtime.GOMAXPROCS(1)
		for n := threads; n > 0; n-- {
			go func() {
				for {
					_ = md5.Sum([]byte("Hello, !"))
				}
			}()
		}
		runtime.GOMAXPROCS(threads)
		argv := append(os.Args, "-test.run=^$")
		if err := syscall.Exec(testenv.Executable(t), argv, os.Environ()); err != nil {
			t.Fatal(err)
		}
	}

	var cmds []*exec.Cmd
	defer func() {
		for _, cmd := range cmds {
			cmd.Process.Kill()
		}
	}()

	args := append(append([]string(nil), os.Args[1:]...), "-test.run=^Test18146$")
	for n := attempts; n > 0; n-- {
		cmd := exec.Command(testenv.Executable(t), args...)
		cmd.Env = append(os.Environ(), "test18146=exec")
		buf := bytes.NewBuffer(nil)
		cmd.Stdout = buf
		cmd.Stderr = buf
		if err := cmd.Start(); err != nil {
			// We are starting so many processes that on
			// some systems (problem seen on Darwin,
			// Dragonfly, OpenBSD) the fork call will fail
			// with EAGAIN.
			if pe, ok := err.(*os.PathError); ok {
				err = pe.Err
			}
			if se, ok := err.(syscall.Errno); ok && (se == syscall.EAGAIN || se == syscall.EMFILE) {
				time.Sleep(time.Millisecond)
				continue
			}

			t.Error(err)
			return
		}
		cmds = append(cmds, cmd)
	}

	failures := 0
	for _, cmd := range cmds {
		err := cmd.Wait()
		if err == nil {
			continue
		}

		t.Errorf("syscall.Exec failed: %v\n%s", err, cmd.Stdout)
		failures++
	}

	if failures > 0 {
		t.Logf("Failed %v of %v attempts.", failures, len(cmds))
	}
}
