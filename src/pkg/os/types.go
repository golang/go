// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

// An operating-system independent representation of Unix data structures.
// OS-specific routines in this directory convert the OS-local versions to these.

// Getpagesize returns the underlying system's memory page size.
func Getpagesize() int { return syscall.Getpagesize() }

// A FileInfo describes a file and is returned by Stat, Fstat, and Lstat
type FileInfo struct {
	Dev             uint64 // device number of file system holding file.
	Ino             uint64 // inode number.
	Nlink           uint64 // number of hard links.
	Mode            uint32 // permission and mode bits.
	Uid             int    // user id of owner.
	Gid             int    // group id of owner.
	Rdev            uint64 // device type for special file.
	Size            int64  // length in bytes.
	Blksize         int64  // size of blocks, in bytes.
	Blocks          int64  // number of blocks allocated for file.
	Atime_ns        int64  // access time; nanoseconds since epoch.
	Mtime_ns        int64  // modified time; nanoseconds since epoch.
	Ctime_ns        int64  // status change time; nanoseconds since epoch.
	Name            string // name of file as presented to Open.
	FollowedSymlink bool   // followed a symlink to get this information
}

// IsFifo reports whether the FileInfo describes a FIFO file.
func (f *FileInfo) IsFifo() bool { return (f.Mode & syscall.S_IFMT) == syscall.S_IFIFO }

// IsChar reports whether the FileInfo describes a character special file.
func (f *FileInfo) IsChar() bool { return (f.Mode & syscall.S_IFMT) == syscall.S_IFCHR }

// IsDirectory reports whether the FileInfo describes a directory.
func (f *FileInfo) IsDirectory() bool { return (f.Mode & syscall.S_IFMT) == syscall.S_IFDIR }

// IsBlock reports whether the FileInfo describes a block special file.
func (f *FileInfo) IsBlock() bool { return (f.Mode & syscall.S_IFMT) == syscall.S_IFBLK }

// IsRegular reports whether the FileInfo describes a regular file.
func (f *FileInfo) IsRegular() bool { return (f.Mode & syscall.S_IFMT) == syscall.S_IFREG }

// IsSymlink reports whether the FileInfo describes a symbolic link.
func (f *FileInfo) IsSymlink() bool { return (f.Mode & syscall.S_IFMT) == syscall.S_IFLNK }

// IsSocket reports whether the FileInfo describes a socket.
func (f *FileInfo) IsSocket() bool { return (f.Mode & syscall.S_IFMT) == syscall.S_IFSOCK }

// Permission returns the file permission bits.
func (f *FileInfo) Permission() uint32 { return f.Mode & 0777 }
