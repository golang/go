// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

// An operating-system independent representation of Unix data structures.
// OS-specific routines in this directory convert the OS-local versions to these.

// Getpagesize returns the underlying system's memory page size.
func Getpagesize() int{
	return syscall.Getpagesize()
}

// A Dir describes a file and is returned by Stat, Fstat, and Lstat
type Dir struct {
	Dev	uint64;	// device number of file system holding file.
	Ino	uint64;	// inode number.
	Nlink	uint64;	// number of hard links.
	Mode	uint32;	// permission and mode bits.
	Uid	uint32;	// user id of owner.
	Gid	uint32;	// group id of owner.
	Rdev	uint64;	// device type for special file.
	Size	uint64;	// length in bytes.
	Blksize	uint64;	// size of blocks, in bytes.
	Blocks	uint64;	// number of blocks allocated for file.
	Atime_ns	uint64;	// access time; nanoseconds since epoch.
	Mtime_ns	uint64;	// modified time; nanoseconds since epoch.
	Ctime_ns	uint64;	// status change time; nanoseconds since epoch.
	Name	string;	// name of file as presented to Open.
	FollowedSymlink	bool;		// followed a symlink to get this information
}

// IsFifo reports whether the Dir describes a FIFO file.
func (dir *Dir) IsFifo() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFIFO
}

// IsChar reports whether the Dir describes a character special file.
func (dir *Dir) IsChar() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFCHR
}

// IsDirectory reports whether the Dir describes a directory.
func (dir *Dir) IsDirectory() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFDIR
}

// IsBlock reports whether the Dir describes a block special file.
func (dir *Dir) IsBlock() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFBLK
}

// IsRegular reports whether the Dir describes a regular file.
func (dir *Dir) IsRegular() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFREG
}

// IsSymlink reports whether the Dir describes a symbolic link.
func (dir *Dir) IsSymlink() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFLNK
}

// IsSocket reports whether the Dir describes a socket.
func (dir *Dir) IsSocket() bool {
	return (dir.Mode & syscall.S_IFMT) == syscall.S_IFSOCK
}

// Permission returns the file permission bits.
func (dir *Dir) Permission() int {
	return int(dir.Mode & 0777)
}

