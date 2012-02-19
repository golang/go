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

// readFile reads and returns the content of the named file.
// It is a trivial implementation of ioutil.ReadFile, reimplemented
// here to avoid depending on io/ioutil or os.
func readFile(name string) ([]byte, error) {
	f, err := syscall.Open(name, syscall.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	defer syscall.Close(f)
	var (
		buf [4096]byte
		ret []byte
		n   int
	)
	for {
		n, err = syscall.Read(f, buf[:])
		if n > 0 {
			ret = append(ret, buf[:n]...)
		}
		if n == 0 || err != nil {
			break
		}
	}
	return ret, err
}

func open(name string) (uintptr, error) {
	fd, err := syscall.Open(name, syscall.O_RDONLY, 0)
	if err != nil {
		return 0, err
	}
	return uintptr(fd), nil
}

func closefd(fd uintptr) {
	syscall.Close(syscall.Handle(fd))
}

func preadn(fd uintptr, buf []byte, off int) error {
	whence := 0
	if off < 0 {
		whence = 2
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
