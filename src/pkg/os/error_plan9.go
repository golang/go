// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"errors"
	"syscall"
)

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
