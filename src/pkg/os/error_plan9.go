// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import syscall "syscall"

// SyscallError records an error from a specific system call.
type SyscallError struct {
	Syscall string
	Err     string
}

func (e *SyscallError) String() string { return e.Syscall + ": " + e.Err }

// Note: If the name of the function NewSyscallError changes,
// pkg/go/doc/doc.go should be adjusted since it hardwires
// this name in a heuristic.

// NewSyscallError returns, as an Error, a new SyscallError
// with the given system call name and error details.
// As a convenience, if err is nil, NewSyscallError returns nil.
func NewSyscallError(syscall string, err syscall.Error) Error {
	if err == nil {
		return nil
	}
	return &SyscallError{syscall, err.String()}
}

var (
	Eshortstat = NewError("stat buffer too small")
	Ebadstat   = NewError("malformed stat buffer")
	Ebadfd     = NewError("fd out of range or not open")
	Ebadarg    = NewError("bad arg in system call")
	Enotdir    = NewError("not a directory")
	Enonexist  = NewError("file does not exist")
	Eexist     = NewError("file already exists")
	Eio        = NewError("i/o error")
	Eperm      = NewError("permission denied")

	EINVAL  = Ebadarg
	ENOTDIR = Enotdir
	ENOENT  = Enonexist
	EEXIST  = Eexist
	EIO     = Eio
	EACCES  = Eperm
	EPERM   = Eperm
	EISDIR  = syscall.EISDIR

	EBADF        = NewError("bad file descriptor")
	ENAMETOOLONG = NewError("file name too long")
	ERANGE       = NewError("math result not representable")
	EPIPE        = NewError("Broken Pipe")
	EPLAN9       = NewError("not supported by plan 9")
)

func iserror(err syscall.Error) bool {
	return err != nil
}

func Errno(e syscall.Error) syscall.Error { return e }
