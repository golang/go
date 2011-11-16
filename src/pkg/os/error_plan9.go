// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"errors"
	"syscall"
)

// SyscallError records an error from a specific system call.
type SyscallError struct {
	Syscall string
	Err     string
}

func (e *SyscallError) Error() string { return e.Syscall + ": " + e.Err }

// Note: If the name of the function NewSyscallError changes,
// pkg/go/doc/doc.go should be adjusted since it hardwires
// this name in a heuristic.

// NewSyscallError returns, as an error, a new SyscallError
// with the given system call name and error details.
// As a convenience, if err is nil, NewSyscallError returns nil.
func NewSyscallError(syscall string, err error) error {
	if err == nil {
		return nil
	}
	return &SyscallError{syscall, err.Error()}
}

var (
	Eshortstat = errors.New("stat buffer too small")
	Ebadstat   = errors.New("malformed stat buffer")
	Ebadfd     = errors.New("fd out of range or not open")
	Ebadarg    = errors.New("bad arg in system call")
	Enotdir    = errors.New("not a directory")
	Enonexist  = errors.New("file does not exist")
	Eexist     = errors.New("file already exists")
	Eio        = errors.New("i/o error")
	Eperm      = errors.New("permission denied")

	EINVAL  = Ebadarg
	ENOTDIR = Enotdir
	ENOENT  = Enonexist
	EEXIST  = Eexist
	EIO     = Eio
	EACCES  = Eperm
	EPERM   = Eperm
	EISDIR  = syscall.EISDIR

	EBADF        = errors.New("bad file descriptor")
	ENAMETOOLONG = errors.New("file name too long")
	ERANGE       = errors.New("math result not representable")
	EPIPE        = errors.New("Broken Pipe")
	EPLAN9       = errors.New("not supported by plan 9")
)
