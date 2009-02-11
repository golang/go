// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

// An operating-system independent representation of Unix data structures.
// OS-specific routines in this directory convert the OS-local versions to these.

// Result of stat64(2) etc.
type Dir struct {
	Dev	uint64;
	Ino	uint64;
	Nlink	uint64;
	Mode	uint32;
	Uid	uint32;
	Gid	uint32;
	Rdev	uint64;
	Size	uint64;
	Blksize	uint64;
	Blocks	uint64;
	Atime_ns	uint64;	// nanoseconds since 1970
	Mtime_ns	uint64;	// nanoseconds since 1970
	Ctime_ns	uint64;	// nanoseconds since 1970
	Name	string;
}

func (dir *Dir) IsFifo() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFIFO
}

func (dir *Dir) IsChar() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFCHR
}

func (dir *Dir) IsDirectory() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFDIR
}

func (dir *Dir) IsBlock() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFBLK
}

func (dir *Dir) IsRegular() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFREG
}

func (dir *Dir) IsSymlink() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFLNK
}

func (dir *Dir) IsSocket() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFSOCK
}

func (dir *Dir) Permission() int {
	return int(dir.Mode & 0777)
}
