// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package os provides a platform-independent interface to operating system
// functionality.  The design is Unix-like.
// The os interface is intended to be uniform across all operating systems.
// Features not generally available appear in the system-specific package syscall.
package os

import (
	"io"
	"syscall"
)

// Name returns the name of the file as presented to Open.
func (file *File) Name() string { return file.name }

// Stdin, Stdout, and Stderr are open Files pointing to the standard input,
// standard output, and standard error file descriptors.
var (
	Stdin  = NewFile(syscall.Stdin, "/dev/stdin")
	Stdout = NewFile(syscall.Stdout, "/dev/stdout")
	Stderr = NewFile(syscall.Stderr, "/dev/stderr")
)

// Flags to Open wrapping those of the underlying system. Not all flags
// may be implemented on a given system.
const (
	O_RDONLY   int = syscall.O_RDONLY   // open the file read-only.
	O_WRONLY   int = syscall.O_WRONLY   // open the file write-only.
	O_RDWR     int = syscall.O_RDWR     // open the file read-write.
	O_APPEND   int = syscall.O_APPEND   // append data to the file when writing.
	O_ASYNC    int = syscall.O_ASYNC    // generate a signal when I/O is available.
	O_CREATE   int = syscall.O_CREAT    // create a new file if none exists.
	O_EXCL     int = syscall.O_EXCL     // used with O_CREATE, file must not exist
	O_NOCTTY   int = syscall.O_NOCTTY   // do not make file the controlling tty.
	O_NONBLOCK int = syscall.O_NONBLOCK // open in non-blocking mode.
	O_NDELAY   int = O_NONBLOCK         // synonym for O_NONBLOCK
	O_SYNC     int = syscall.O_SYNC     // open for synchronous I/O.
	O_TRUNC    int = syscall.O_TRUNC    // if possible, truncate file when opened.
)

// Seek whence values.
const (
	SEEK_SET int = 0 // seek relative to the origin of the file
	SEEK_CUR int = 1 // seek relative to the current offset
	SEEK_END int = 2 // seek relative to the end
)

// Read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
// EOF is signaled by a zero count with err set to io.EOF.
func (file *File) Read(b []byte) (n int, err error) {
	if file == nil {
		return 0, EINVAL
	}
	n, e := file.read(b)
	if n < 0 {
		n = 0
	}
	if n == 0 && len(b) > 0 && e == nil {
		return 0, io.EOF
	}
	if e != nil {
		err = &PathError{"read", file.name, e}
	}
	return n, err
}

// ReadAt reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to io.EOF.
// ReadAt always returns a non-nil error when n != len(b).
func (file *File) ReadAt(b []byte, off int64) (n int, err error) {
	if file == nil {
		return 0, EINVAL
	}
	for len(b) > 0 {
		m, e := file.pread(b, off)
		if m == 0 && e == nil {
			return n, io.EOF
		}
		if e != nil {
			err = &PathError{"read", file.name, e}
			break
		}
		n += m
		b = b[m:]
		off += int64(m)
	}
	return
}

// Write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
// Write returns a non-nil error when n != len(b).
func (file *File) Write(b []byte) (n int, err error) {
	if file == nil {
		return 0, EINVAL
	}
	n, e := file.write(b)
	if n < 0 {
		n = 0
	}

	epipecheck(file, e)

	if e != nil {
		err = &PathError{"write", file.name, e}
	}
	return n, err
}

// WriteAt writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
// WriteAt returns a non-nil error when n != len(b).
func (file *File) WriteAt(b []byte, off int64) (n int, err error) {
	if file == nil {
		return 0, EINVAL
	}
	for len(b) > 0 {
		m, e := file.pwrite(b, off)
		if e != nil {
			err = &PathError{"write", file.name, e}
			break
		}
		n += m
		b = b[m:]
		off += int64(m)
	}
	return
}

// Seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an error, if any.
func (file *File) Seek(offset int64, whence int) (ret int64, err error) {
	r, e := file.seek(offset, whence)
	if e == nil && file.dirinfo != nil && r != 0 {
		e = syscall.EISDIR
	}
	if e != nil {
		return 0, &PathError{"seek", file.name, e}
	}
	return r, nil
}

// WriteString is like Write, but writes the contents of string s rather than
// an array of bytes.
func (file *File) WriteString(s string) (ret int, err error) {
	if file == nil {
		return 0, EINVAL
	}
	return file.Write([]byte(s))
}

// Mkdir creates a new directory with the specified name and permission bits.
// It returns an error, if any.
func Mkdir(name string, perm uint32) error {
	e := syscall.Mkdir(name, perm)
	if e != nil {
		return &PathError{"mkdir", name, e}
	}
	return nil
}

// Chdir changes the current working directory to the named directory.
func Chdir(dir string) error {
	if e := syscall.Chdir(dir); e != nil {
		return &PathError{"chdir", dir, e}
	}
	return nil
}

// Chdir changes the current working directory to the file,
// which must be a directory.
func (f *File) Chdir() error {
	if e := syscall.Fchdir(f.fd); e != nil {
		return &PathError{"chdir", f.name, e}
	}
	return nil
}

// Open opens the named file for reading.  If successful, methods on
// the returned file can be used for reading; the associated file
// descriptor has mode O_RDONLY.
// It returns the File and an error, if any.
func Open(name string) (file *File, err error) {
	return OpenFile(name, O_RDONLY, 0)
}

// Create creates the named file mode 0666 (before umask), truncating
// it if it already exists.  If successful, methods on the returned
// File can be used for I/O; the associated file descriptor has mode
// O_RDWR.
// It returns the File and an error, if any.
func Create(name string) (file *File, err error) {
	return OpenFile(name, O_RDWR|O_CREATE|O_TRUNC, 0666)
}
