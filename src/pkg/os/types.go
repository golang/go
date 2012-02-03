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
	Sys() interface{}   // underlying data source (can return nil)
}

// A FileMode represents a file's mode and permission bits.
// The bits have the same definition on all systems, so that
// information about files can be moved from one system
// to another portably.  Not all bits apply to all systems.
// The only required bit is ModeDir for directories.
type FileMode uint32

// The defined file mode bits are the most significant bits of the FileMode.
// The nine least-significant bits are the standard Unix rwxrwxrwx permissions.
// The values of these bits should be considered part of the public API and
// may be used in wire protocols or disk representations: they must not be
// changed, although new bits might be added.
const (
	// The single letters are the abbreviations
	// used by the String method's formatting.
	ModeDir        FileMode = 1 << (32 - 1 - iota) // d: is a directory
	ModeAppend                                     // a: append-only
	ModeExclusive                                  // l: exclusive use
	ModeTemporary                                  // T: temporary file (not backed up)
	ModeSymlink                                    // L: symbolic link
	ModeDevice                                     // D: device file
	ModeNamedPipe                                  // p: named pipe (FIFO)
	ModeSocket                                     // S: Unix domain socket
	ModeSetuid                                     // u: setuid
	ModeSetgid                                     // g: setgid
	ModeCharDevice                                 // c: Unix character device, when ModeDevice is set
	ModeSticky                                     // t: sticky

	// Mask for the type bits. For regular files, none will be set.
	ModeType = ModeDir | ModeSymlink | ModeNamedPipe | ModeSocket | ModeDevice

	ModePerm FileMode = 0777 // permission bits
)

func (m FileMode) String() string {
	const str = "dalTLDpSugct"
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

// A fileStat is the implementation of FileInfo returned by Stat and Lstat.
type fileStat struct {
	name    string
	size    int64
	mode    FileMode
	modTime time.Time
	sys     interface{}
}

func (fs *fileStat) Name() string       { return fs.name }
func (fs *fileStat) Size() int64        { return fs.size }
func (fs *fileStat) Mode() FileMode     { return fs.mode }
func (fs *fileStat) ModTime() time.Time { return fs.modTime }
func (fs *fileStat) IsDir() bool        { return fs.mode.IsDir() }
func (fs *fileStat) Sys() interface{}   { return fs.sys }

// SameFile reports whether fi1 and fi2 describe the same file.
// For example, on Unix this means that the device and inode fields
// of the two underlying structures are identical; on other systems
// the decision may be based on the path names.
// SameFile only applies to results returned by this package's Stat.
// It returns false in other cases.
func SameFile(fi1, fi2 FileInfo) bool {
	fs1, ok1 := fi1.(*fileStat)
	fs2, ok2 := fi2.(*fileStat)
	if !ok1 || !ok2 {
		return false
	}
	return sameFile(fs1.sys, fs2.sys)
}
