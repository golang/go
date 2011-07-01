// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package os provides a platform-independent interface to operating system
// functionality.  The design is Unix-like.
// The os interface is intended to be uniform across all operating systems.
// Features not generally available appear in the system-specific package syscall.
package os

import (
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

type eofError int

func (eofError) String() string { return "EOF" }

// EOF is the Error returned by Read when no more input is available.
// Functions should return EOF only to signal a graceful end of input.
// If the EOF occurs unexpectedly in a structured data stream,
// the appropriate error is either io.ErrUnexpectedEOF or some other error
// giving more detail.
var EOF Error = eofError(0)

// Read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an Error, if any.
// EOF is signaled by a zero count with err set to EOF.
func (file *File) Read(b []byte) (n int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	n, e := file.read(b)
	if n < 0 {
		n = 0
	}
	if n == 0 && !iserror(e) {
		return 0, EOF
	}
	if iserror(e) {
		err = &PathError{"read", file.name, Errno(e)}
	}
	return n, err
}

// ReadAt reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the Error, if any.
// EOF is signaled by a zero count with err set to EOF.
// ReadAt always returns a non-nil Error when n != len(b).
func (file *File) ReadAt(b []byte, off int64) (n int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	for len(b) > 0 {
		m, e := file.pread(b, off)
		if m == 0 && !iserror(e) {
			return n, EOF
		}
		if iserror(e) {
			err = &PathError{"read", file.name, Errno(e)}
			break
		}
		n += m
		b = b[m:]
		off += int64(m)
	}
	return
}

// Write writes len(b) bytes to the File.
// It returns the number of bytes written and an Error, if any.
// Write returns a non-nil Error when n != len(b).
func (file *File) Write(b []byte) (n int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	n, e := file.write(b)
	if n < 0 {
		n = 0
	}

	epipecheck(file, e)

	if iserror(e) {
		err = &PathError{"write", file.name, Errno(e)}
	}
	return n, err
}

// WriteAt writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an Error, if any.
// WriteAt returns a non-nil Error when n != len(b).
func (file *File) WriteAt(b []byte, off int64) (n int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	for len(b) > 0 {
		m, e := file.pwrite(b, off)
		if iserror(e) {
			err = &PathError{"write", file.name, Errno(e)}
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
// It returns the new offset and an Error, if any.
func (file *File) Seek(offset int64, whence int) (ret int64, err Error) {
	r, e := file.seek(offset, whence)
	if !iserror(e) && file.dirinfo != nil && r != 0 {
		e = syscall.EISDIR
	}
	if iserror(e) {
		return 0, &PathError{"seek", file.name, Errno(e)}
	}
	return r, nil
}

// WriteString is like Write, but writes the contents of string s rather than
// an array of bytes.
func (file *File) WriteString(s string) (ret int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	return file.Write([]byte(s))
}

// Mkdir creates a new directory with the specified name and permission bits.
// It returns an error, if any.
func Mkdir(name string, perm uint32) Error {
	e := syscall.Mkdir(name, perm)
	if iserror(e) {
		return &PathError{"mkdir", name, Errno(e)}
	}
	return nil
}

// Chdir changes the current working directory to the named directory.
func Chdir(dir string) Error {
	if e := syscall.Chdir(dir); iserror(e) {
		return &PathError{"chdir", dir, Errno(e)}
	}
	return nil
}

// Chdir changes the current working directory to the file,
// which must be a directory.
func (f *File) Chdir() Error {
	if e := syscall.Fchdir(f.fd); iserror(e) {
		return &PathError{"chdir", f.name, Errno(e)}
	}
	return nil
}

// Open opens the named file for reading.  If successful, methods on
// the returned file can be used for reading; the associated file
// descriptor has mode O_RDONLY.
// It returns the File and an Error, if any.
func Open(name string) (file *File, err Error) {
	return OpenFile(name, O_RDONLY, 0)
}

// Create creates the named file mode 0666 (before umask), truncating
// it if it already exists.  If successful, methods on the returned
// File can be used for I/O; the associated file descriptor has mode
// O_RDWR.
// It returns the File and an Error, if any.
func Create(name string) (file *File, err Error) {
	return OpenFile(name, O_RDWR|O_CREATE|O_TRUNC, 0666)
}
