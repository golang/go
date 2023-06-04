// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm)

package poll

import "syscall"

type SysFile struct {
	// Writev cache.
	iovecs *[]syscall.Iovec
}

func (s *SysFile) init() {}

func (s *SysFile) destroy(fd int) error {
	// We don't use ignoringEINTR here because POSIX does not define
	// whether the descriptor is closed if close returns EINTR.
	// If the descriptor is indeed closed, using a loop would race
	// with some other goroutine opening a new descriptor.
	// (The Linux kernel guarantees that it is closed on an EINTR error.)
	return CloseFunc(fd)
}

// dupCloseOnExecOld is the traditional way to dup an fd and
// set its O_CLOEXEC bit, using two system calls.
func dupCloseOnExecOld(fd int) (int, string, error) {
	syscall.ForkLock.RLock()
	defer syscall.ForkLock.RUnlock()
	newfd, err := syscall.Dup(fd)
	if err != nil {
		return -1, "dup", err
	}
	syscall.CloseOnExec(newfd)
	return newfd, "", nil
}

// Fchdir wraps syscall.Fchdir.
func (fd *FD) Fchdir() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Fchdir(fd.Sysfd)
}

// ReadDirent wraps syscall.ReadDirent.
// We treat this like an ordinary system call rather than a call
// that tries to fill the buffer.
func (fd *FD) ReadDirent(buf []byte) (int, error) {
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()
	for {
		n, err := ignoringEINTRIO(syscall.ReadDirent, fd.Sysfd, buf)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		// Do not call eofError; caller does not expect to see io.EOF.
		return n, err
	}
}

// Seek wraps syscall.Seek.
func (fd *FD) Seek(offset int64, whence int) (int64, error) {
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()
	return syscall.Seek(fd.Sysfd, offset, whence)
}
