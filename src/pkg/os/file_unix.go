// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The os package provides a platform-independent interface to operating
// system functionality.  The design is Unix-like.
package os

import (
	"runtime"
	"syscall"
)

// Auxiliary information if the File describes a directory
type dirInfo struct {
	buf  []byte // buffer for directory I/O
	nbuf int    // length of buf; return value from Getdirentries
	bufp int    // location of next record in buf.
}

// DevNull is the name of the operating system's ``null device.''
// On Unix-like systems, it is "/dev/null"; on Windows, "NUL".
const DevNull = "/dev/null"

// Open opens the named file with specified flag (O_RDONLY etc.) and perm, (0666 etc.)
// if applicable.  If successful, methods on the returned File can be used for I/O.
// It returns the File and an Error, if any.
func Open(name string, flag int, perm uint32) (file *File, err Error) {
	r, e := syscall.Open(name, flag|syscall.O_CLOEXEC, perm)
	if e != 0 {
		return nil, &PathError{"open", name, Errno(e)}
	}

	// There's a race here with fork/exec, which we are
	// content to live with.  See ../syscall/exec.go
	if syscall.O_CLOEXEC == 0 { // O_CLOEXEC not supported
		syscall.CloseOnExec(r)
	}

	return NewFile(r, name), nil
}

// Close closes the File, rendering it unusable for I/O.
// It returns an Error, if any.
func (file *File) Close() Error {
	if file == nil || file.fd < 0 {
		return EINVAL
	}
	var err Error
	if e := syscall.Close(file.fd); e != 0 {
		err = &PathError{"close", file.name, Errno(e)}
	}
	file.fd = -1 // so it can't be closed again

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

// Stat returns the FileInfo structure describing file.
// It returns the FileInfo and an error, if any.
func (file *File) Stat() (fi *FileInfo, err Error) {
	var stat syscall.Stat_t
	e := syscall.Fstat(file.fd, &stat)
	if e != 0 {
		return nil, &PathError{"stat", file.name, Errno(e)}
	}
	return fileInfoFromStat(file.name, new(FileInfo), &stat, &stat), nil
}

// Readdir reads the contents of the directory associated with file and
// returns an array of up to count FileInfo structures, as would be returned
// by Stat, in directory order.  Subsequent calls on the same file will yield
// further FileInfos.
// A negative count means to read until EOF.
// Readdir returns the array and an Error, if any.
func (file *File) Readdir(count int) (fi []FileInfo, err Error) {
	dirname := file.name
	if dirname == "" {
		dirname = "."
	}
	dirname += "/"
	names, err1 := file.Readdirnames(count)
	if err1 != nil {
		return nil, err1
	}
	fi = make([]FileInfo, len(names))
	for i, filename := range names {
		fip, err := Lstat(dirname + filename)
		if fip == nil || err != nil {
			fi[i].Name = filename // rest is already zeroed out
		} else {
			fi[i] = *fip
		}
	}
	return
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
func Truncate(name string, size int64) Error {
	if e := syscall.Truncate(name, size); e != 0 {
		return &PathError{"truncate", name, Errno(e)}
	}
	return nil
}
