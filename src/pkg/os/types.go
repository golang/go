// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"time"
)

// Getpagesize returns the underlying system's memory page size.
func Getpagesize() int { return syscall.Getpagesize() }

// A FileInfo describes a file and is returned by Stat and Lstat
type FileInfo interface {
	Name() string       // base name of the file
	Size() int64        // length in bytes
	Mode() FileMode     // file mode bits
	ModTime() time.Time // modification time
	IsDir() bool        // abbreviation for Mode().IsDir()
}

// A FileMode represents a file's mode and permission bits.
// The bits have the same definition on all systems, so that
// information about files can be moved from one system
// to another portably.  Not all bits apply to all systems.
// The only required bit is ModeDir for directories.
type FileMode uint32

// The defined file mode bits are the most significant bits of the FileMode.
// The nine least-significant bits are the standard Unix rwxrwxrwx permissions.
const (
	// The single letters are the abbreviations
	// used by the String method's formatting.
	ModeDir       FileMode = 1 << (32 - 1 - iota) // d: is a directory
	ModeAppend                                    // a: append-only
	ModeExclusive                                 // l: exclusive use
	ModeTemporary                                 // t: temporary file (not backed up)
	ModeSymlink                                   // L: symbolic link
	ModeDevice                                    // D: device file
	ModeNamedPipe                                 // p: named pipe (FIFO)
	ModeSocket                                    // S: Unix domain socket
	ModeSetuid                                    // u: setuid
	ModeSetgid                                    // g: setgid

	// Mask for the type bits. For regular files, none will be set.
	ModeType = ModeDir | ModeSymlink | ModeNamedPipe | ModeSocket | ModeDevice

	ModePerm FileMode = 0777 // permission bits
)

func (m FileMode) String() string {
	const str = "daltLDpSug"
	var buf [20]byte
	w := 0
	for i, c := range str {
		if m&(1<<uint(32-1-i)) != 0 {
			buf[w] = byte(c)
			w++
		}
	}
	if w == 0 {
		buf[w] = '-'
		w++
	}
	const rwx = "rwxrwxrwx"
	for i, c := range rwx {
		if m&(1<<uint(9-1-i)) != 0 {
			buf[w] = byte(c)
		} else {
			buf[w] = '-'
		}
		w++
	}
	return string(buf[:w])
}

// IsDir reports whether m describes a directory.
// That is, it tests for the ModeDir bit being set in m.
func (m FileMode) IsDir() bool {
	return m&ModeDir != 0
}

// Perm returns the Unix permission bits in m.
func (m FileMode) Perm() FileMode {
	return m & ModePerm
}

// A FileStat is the implementation of FileInfo returned by Stat and Lstat.
// Clients that need access to the underlying system-specific stat information
// can test for *os.FileStat and then consult the Sys field.
type FileStat struct {
	name    string
	size    int64
	mode    FileMode
	modTime time.Time

	Sys interface{}
}

func (fs *FileStat) Name() string       { return fs.name }
func (fs *FileStat) Size() int64        { return fs.size }
func (fs *FileStat) Mode() FileMode     { return fs.mode }
func (fs *FileStat) ModTime() time.Time { return fs.modTime }
func (fs *FileStat) IsDir() bool        { return fs.mode.IsDir() }

// SameFile reports whether fs and other describe the same file.
// For example, on Unix this means that the device and inode fields
// of the two underlying structures are identical; on other systems
// the decision may be based on the path names.
func (fs *FileStat) SameFile(other *FileStat) bool {
	return sameFile(fs, other)
}
