// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package os

import (
	"runtime"
	"syscall"
)

// File represents an open file descriptor.
type File struct {
	*file
}

// file is the real representation of *File.
// The extra level of indirection ensures that no clients of os
// can overwrite this data, which could cause the finalizer
// to close the wrong file descriptor.
type file struct {
	fd      int
	name    string
	dirinfo *dirInfo // nil unless directory being read
	nepipe  int      // number of consecutive EPIPE in Write
}

// Fd returns the integer Unix file descriptor referencing the open file.
func (f *File) Fd() int {
	if f == nil {
		return -1
	}
	return f.fd
}

// NewFile returns a new File with the given file descriptor and name.
func NewFile(fd int, name string) *File {
	if fd < 0 {
		return nil
	}
	f := &File{&file{fd: fd, name: name}}
	runtime.SetFinalizer(f.file, (*file).close)
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
	if e != nil {
		return nil, &PathError{"open", name, e}
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
func (f *File) Close() error {
	return f.file.close()
}

func (file *file) close() error {
	if file == nil || file.fd < 0 {
		return EINVAL
	}
	var err error
	if e := syscall.Close(file.fd); e != nil {
		err = &PathError{"close", file.name, e}
	}
	file.fd = -1 // so it can't be closed again

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

// Stat returns the FileInfo structure describing file.
// It returns the FileInfo and an error, if any.
func (f *File) Stat() (fi FileInfo, err error) {
	var stat syscall.Stat_t
	err = syscall.Fstat(f.fd, &stat)
	if err != nil {
		return nil, &PathError{"stat", f.name, err}
	}
	return fileInfoFromStat(&stat, f.name), nil
}

// Stat returns a FileInfo describing the named file and an error, if any.
// If name names a valid symbolic link, the returned FileInfo describes
// the file pointed at by the link and has fi.FollowedSymlink set to true.
// If name names an invalid symbolic link, the returned FileInfo describes
// the link itself and has fi.FollowedSymlink set to false.
func Stat(name string) (fi FileInfo, err error) {
	var stat syscall.Stat_t
	err = syscall.Stat(name, &stat)
	if err != nil {
		return nil, &PathError{"stat", name, err}
	}
	return fileInfoFromStat(&stat, name), nil
}

// Lstat returns a FileInfo describing the named file and an
// error, if any.  If the file is a symbolic link, the returned FileInfo
// describes the symbolic link.  Lstat makes no attempt to follow the link.
func Lstat(name string) (fi FileInfo, err error) {
	var stat syscall.Stat_t
	err = syscall.Lstat(name, &stat)
	if err != nil {
		return nil, &PathError{"lstat", name, err}
	}
	return fileInfoFromStat(&stat, name), nil
}

// Readdir reads the contents of the directory associated with file and
// returns an array of up to n FileInfo values, as would be returned
// by Lstat, in directory order. Subsequent calls on the same file will yield
// further FileInfos.
//
// If n > 0, Readdir returns at most n FileInfo structures. In this case, if
// Readdir returns an empty slice, it will return a non-nil error
// explaining why. At the end of a directory, the error is io.EOF.
//
// If n <= 0, Readdir returns all the FileInfo from the directory in
// a single slice. In this case, if Readdir succeeds (reads all
// the way to the end of the directory), it returns the slice and a
// nil error. If it encounters an error before the end of the
// directory, Readdir returns the FileInfo read until that point
// and a non-nil error.
func (f *File) Readdir(n int) (fi []FileInfo, err error) {
	dirname := f.name
	if dirname == "" {
		dirname = "."
	}
	dirname += "/"
	names, err := f.Readdirnames(n)
	fi = make([]FileInfo, len(names))
	for i, filename := range names {
		fip, err := Lstat(dirname + filename)
		if err == nil {
			fi[i] = fip
		} else {
			fi[i] = &FileStat{name: filename}
		}
	}
	return fi, err
}

// read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
func (f *File) read(b []byte) (n int, err error) {
	return syscall.Read(f.fd, b)
}

// pread reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to 0.
func (f *File) pread(b []byte, off int64) (n int, err error) {
	return syscall.Pread(f.fd, b, off)
}

// write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
func (f *File) write(b []byte) (n int, err error) {
	return syscall.Write(f.fd, b)
}

// pwrite writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
func (f *File) pwrite(b []byte, off int64) (n int, err error) {
	return syscall.Pwrite(f.fd, b, off)
}

// seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an error, if any.
func (f *File) seek(offset int64, whence int) (ret int64, err error) {
	return syscall.Seek(f.fd, offset, whence)
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
func Truncate(name string, size int64) error {
	if e := syscall.Truncate(name, size); e != nil {
		return &PathError{"truncate", name, e}
	}
	return nil
}

// Remove removes the named file or directory.
func Remove(name string) error {
	// System call interface forces us to know
	// whether name is a file or directory.
	// Try both: it is cheaper on average than
	// doing a Stat plus the right one.
	e := syscall.Unlink(name)
	if e == nil {
		return nil
	}
	e1 := syscall.Rmdir(name)
	if e1 == nil {
		return nil
	}

	// Both failed: figure out which error to return.
	// OS X and Linux differ on whether unlink(dir)
	// returns EISDIR, so can't use that.  However,
	// both agree that rmdir(file) returns ENOTDIR,
	// so we can use that to decide which error is real.
	// Rmdir might also return ENOTDIR if given a bad
	// file path, like /etc/passwd/foo, but in that case,
	// both errors will be ENOTDIR, so it's okay to
	// use the error from unlink.
	if e1 != syscall.ENOTDIR {
		e = e1
	}
	return &PathError{"remove", name, e}
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
	if e != nil {
		syscall.ForkLock.RUnlock()
		return nil, nil, NewSyscallError("pipe", e)
	}
	syscall.CloseOnExec(p[0])
	syscall.CloseOnExec(p[1])
	syscall.ForkLock.RUnlock()

	return NewFile(p[0], "|0"), NewFile(p[1], "|1"), nil
}

// TempDir returns the default directory to use for temporary files.
func TempDir() string {
	dir := Getenv("TMPDIR")
	if dir == "" {
		dir = "/tmp"
	}
	return dir
}
