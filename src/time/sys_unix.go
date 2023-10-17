// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package time

import (
	"errors"
	"runtime"
	"syscall"
)

// for testing: whatever interrupts a sleep
func interrupt() {
	// There is no mechanism in wasi to interrupt the call to poll_oneoff
	// used to implement runtime.usleep so this function does nothing, which
	// somewhat defeats the purpose of TestSleep but we are still better off
	// validating that time elapses when the process calls time.Sleep than
	// skipping the test altogether.
	if runtime.GOOS != "wasip1" {
		syscall.Kill(syscall.Getpid(), syscall.SIGCHLD)
	}
}

func open(name string) (uintptr, error) {
	fd, err := syscall.Open(name, syscall.O_RDONLY, 0)
	if err != nil {
		return 0, err
	}
	return uintptr(fd), nil
}

func read(fd uintptr, buf []byte) (int, error) {
	return syscall.Read(int(fd), buf)
}

func closefd(fd uintptr) {
	syscall.Close(int(fd))
}

func preadn(fd uintptr, buf []byte, off int) error {
	whence := seekStart
	if off < 0 {
		whence = seekEnd
	}
	if _, err := syscall.Seek(int(fd), int64(off), whence); err != nil {
		return err
	}
	for len(buf) > 0 {
		m, err := syscall.Read(int(fd), buf)
		if m <= 0 {
			if err == nil {
				return errors.New("short read")
			}
			return err
		}
		buf = buf[m:]
	}
	return nil
}
