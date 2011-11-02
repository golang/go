// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

package os

import (
	"runtime"
	"syscall"
)

// File represents an open file descriptor.
type File struct {
	fd      int
	name    string
	dirinfo *dirInfo // nil unless directory being read
	nepipe  int      // number of consecutive EPIPE in Write
}

// Fd returns the integer Unix file descriptor referencing the open file.
func (file *File) Fd() int {
	if file == nil {
		return -1
	}
	return file.fd
}

// NewFile returns a new File with the given file descriptor and name.
func NewFile(fd int, name string) *File {
	if fd < 0 {
		return nil
	}
	f := &File{fd: fd, name: name}
	runtime.SetFinalizer(f, (*File).Close)
	return f
}

// Auxiliary information if the File describes a directory
type dirInfo struct {
	buf  []byte // buffer for directory I/O
	nbuf int    // length of buf; return value from Getdirentries
	bufp int    // location of next record in buf.
}

// DevNull is the name of the operating system's ``null device.''
// On Unix-like systems, it is "/dev/null"; on Windows, "NUL".
const DevNull = "/dev/null"

// OpenFile is the generalized open call; most users will use Open
// or Create instead.  It opens the named file with specified flag
// (O_RDONLY etc.) and perm, (0666 etc.) if applicable.  If successful,
// methods on the returned File can be used for I/O.
// It returns the File and an error, if any.
func OpenFile(name string, flag int, perm uint32) (file *File, err error) {
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
// It returns an error, if any.
func (file *File) Close() error {
	if file == nil || file.fd < 0 {
		return EINVAL
	}
	var err error
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
func (file *File) Stat() (fi *FileInfo, err error) {
	var stat syscall.Stat_t
	e := syscall.Fstat(file.fd, &stat)
	if e != 0 {
		return nil, &PathError{"stat", file.name, Errno(e)}
	}
	return fileInfoFromStat(file.name, new(FileInfo), &stat, &stat), nil
}

// Stat returns a FileInfo structure describing the named file and an error, if any.
// If name names a valid symbolic link, the returned FileInfo describes
// the file pointed at by the link and has fi.FollowedSymlink set to true.
// If name names an invalid symbolic link, the returned FileInfo describes
// the link itself and has fi.FollowedSymlink set to false.
func Stat(name string) (fi *FileInfo, err error) {
	var lstat, stat syscall.Stat_t
	e := syscall.Lstat(name, &lstat)
	if iserror(e) {
		return nil, &PathError{"stat", name, Errno(e)}
	}
	statp := &lstat
	if lstat.Mode&syscall.S_IFMT == syscall.S_IFLNK {
		e := syscall.Stat(name, &stat)
		if !iserror(e) {
			statp = &stat
		}
	}
	return fileInfoFromStat(name, new(FileInfo), &lstat, statp), nil
}

// Lstat returns the FileInfo structure describing the named file and an
// error, if any.  If the file is a symbolic link, the returned FileInfo
// describes the symbolic link.  Lstat makes no attempt to follow the link.
func Lstat(name string) (fi *FileInfo, err error) {
	var stat syscall.Stat_t
	e := syscall.Lstat(name, &stat)
	if iserror(e) {
		return nil, &PathError{"lstat", name, Errno(e)}
	}
	return fileInfoFromStat(name, new(FileInfo), &stat, &stat), nil
}

// Readdir reads the contents of the directory associated with file and
// returns an array of up to n FileInfo structures, as would be returned
// by Lstat, in directory order. Subsequent calls on the same file will yield
// further FileInfos.
//
// If n > 0, Readdir returns at most n FileInfo structures. In this case, if
// Readdir returns an empty slice, it will return a non-nil error
// explaining why. At the end of a directory, the error is os.EOF.
//
// If n <= 0, Readdir returns all the FileInfo from the directory in
// a single slice. In this case, if Readdir succeeds (reads all
// the way to the end of the directory), it returns the slice and a
// nil os.Error. If it encounters an error before the end of the
// directory, Readdir returns the FileInfo read until that point
// and a non-nil error.
func (file *File) Readdir(n int) (fi []FileInfo, err error) {
	dirname := file.name
	if dirname == "" {
		dirname = "."
	}
	dirname += "/"
	names, err := file.Readdirnames(n)
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

// read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
func (f *File) read(b []byte) (n int, err int) {
	return syscall.Read(f.fd, b)
}

// pread reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to 0.
func (f *File) pread(b []byte, off int64) (n int, err int) {
	return syscall.Pread(f.fd, b, off)
}

// write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
func (f *File) write(b []byte) (n int, err int) {
	return syscall.Write(f.fd, b)
}

// pwrite writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
func (f *File) pwrite(b []byte, off int64) (n int, err int) {
	return syscall.Pwrite(f.fd, b, off)
}

// seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an error, if any.
func (f *File) seek(offset int64, whence int) (ret int64, err int) {
	return syscall.Seek(f.fd, offset, whence)
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
func Truncate(name string, size int64) error {
	if e := syscall.Truncate(name, size); e != 0 {
		return &PathError{"truncate", name, Errno(e)}
	}
	return nil
}

// basename removes trailing slashes and the leading directory name from path name
func basename(name string) string {
	i := len(name) - 1
	// Remove trailing slashes
	for ; i > 0 && name[i] == '/'; i-- {
		name = name[:i]
	}
	// Remove leading directory name
	for i--; i >= 0; i-- {
		if name[i] == '/' {
			name = name[i+1:]
			break
		}
	}

	return name
}

// Pipe returns a connected pair of Files; reads from r return bytes written to w.
// It returns the files and an error, if any.
func Pipe() (r *File, w *File, err error) {
	var p [2]int

	// See ../syscall/exec.go for description of lock.
	syscall.ForkLock.RLock()
	e := syscall.Pipe(p[0:])
	if iserror(e) {
		syscall.ForkLock.RUnlock()
		return nil, nil, NewSyscallError("pipe", e)
	}
	syscall.CloseOnExec(p[0])
	syscall.CloseOnExec(p[1])
	syscall.ForkLock.RUnlock()

	return NewFile(p[0], "|0"), NewFile(p[1], "|1"), nil
}
