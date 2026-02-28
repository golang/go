// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plan9

import "syscall"

// Constants
const (
	// Invented values to support what package os expects.
	O_CREAT    = 0x02000
	O_APPEND   = 0x00400
	O_NOCTTY   = 0x00000
	O_NONBLOCK = 0x00000
	O_SYNC     = 0x00000
	O_ASYNC    = 0x00000

	S_IFMT   = 0x1f000
	S_IFIFO  = 0x1000
	S_IFCHR  = 0x2000
	S_IFDIR  = 0x4000
	S_IFBLK  = 0x6000
	S_IFREG  = 0x8000
	S_IFLNK  = 0xa000
	S_IFSOCK = 0xc000
)

// Errors
var (
	EINVAL       = syscall.NewError("bad arg in system call")
	ENOTDIR      = syscall.NewError("not a directory")
	EISDIR       = syscall.NewError("file is a directory")
	ENOENT       = syscall.NewError("file does not exist")
	EEXIST       = syscall.NewError("file already exists")
	EMFILE       = syscall.NewError("no free file descriptors")
	EIO          = syscall.NewError("i/o error")
	ENAMETOOLONG = syscall.NewError("file name too long")
	EINTR        = syscall.NewError("interrupted")
	EPERM        = syscall.NewError("permission denied")
	EBUSY        = syscall.NewError("no free devices")
	ETIMEDOUT    = syscall.NewError("connection timed out")
	EPLAN9       = syscall.NewError("not supported by plan 9")

	// The following errors do not correspond to any
	// Plan 9 system messages. Invented to support
	// what package os and others expect.
	EACCES       = syscall.NewError("access permission denied")
	EAFNOSUPPORT = syscall.NewError("address family not supported by protocol")
)
