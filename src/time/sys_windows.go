// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"errors"
	"syscall"
)

// for testing: whatever interrupts a sleep
func interrupt() {
}

func open(name string) (uintptr, error) {
	fd, err := syscall.Open(name, syscall.O_RDONLY, 0)
	if err != nil {
		// This condition solves issue https://go.dev/issue/50248
		if err == syscall.ERROR_PATH_NOT_FOUND {
			err = syscall.ENOENT
		}
		return 0, err
	}
	return uintptr(fd), nil
}

func read(fd uintptr, buf []byte) (int, error) {
	return syscall.Read(syscall.Handle(fd), buf)
}

func closefd(fd uintptr) {
	syscall.Close(syscall.Handle(fd))
}

func preadn(fd uintptr, buf []byte, off int) error {
	whence := seekStart
	if off < 0 {
		whence = seekEnd
	}
	if _, err := syscall.Seek(syscall.Handle(fd), int64(off), whence); err != nil {
		return err
	}
	for len(buf) > 0 {
		m, err := syscall.Read(syscall.Handle(fd), buf)
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
