// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"io"
	"runtime"
	"sync"
	"syscall"
	"unicode/utf16"
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
	fd      syscall.Handle
	name    string
	dirinfo *dirInfo   // nil unless directory being read
	nepipe  int        // number of consecutive EPIPE in Write
	l       sync.Mutex // used to implement windows pread/pwrite
}

// Fd returns the Windows handle referencing the open file.
func (file *File) Fd() syscall.Handle {
	if file == nil {
		return syscall.InvalidHandle
	}
	return file.fd
}

// NewFile returns a new File with the given file descriptor and name.
func NewFile(fd syscall.Handle, name string) *File {
	if fd < 0 {
		return nil
	}
	f := &File{&file{fd: fd, name: name}}
	runtime.SetFinalizer(f.file, (*file).close)
	return f
}

// Auxiliary information if the File describes a directory
type dirInfo struct {
	data     syscall.Win32finddata
	needdata bool
}

const DevNull = "NUL"

func (file *file) isdir() bool { return file != nil && file.dirinfo != nil }

func openFile(name string, flag int, perm uint32) (file *File, err error) {
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

func openDir(name string) (file *File, err error) {
	d := new(dirInfo)
	r, e := syscall.FindFirstFile(syscall.StringToUTF16Ptr(name+`\*`), &d.data)
	if e != nil {
		return nil, &PathError{"open", name, e}
	}
	f := NewFile(r, name)
	f.dirinfo = d
	return f, nil
}

// OpenFile is the generalized open call; most users will use Open
// or Create instead.  It opens the named file with specified flag
// (O_RDONLY etc.) and perm, (0666 etc.) if applicable.  If successful,
// methods on the returned File can be used for I/O.
// It returns the File and an error, if any.
func OpenFile(name string, flag int, perm uint32) (file *File, err error) {
	if name == "" {
		return nil, &PathError{"open", name, syscall.ENOENT}
	}
	// TODO(brainman): not sure about my logic of assuming it is dir first, then fall back to file
	r, e := openDir(name)
	if e == nil {
		if flag&O_WRONLY != 0 || flag&O_RDWR != 0 {
			r.Close()
			return nil, &PathError{"open", name, EISDIR}
		}
		return r, nil
	}
	r, e = openFile(name, flag, perm)
	if e == nil {
		return r, nil
	}
	return nil, e
}

// Close closes the File, rendering it unusable for I/O.
// It returns an error, if any.
func (file *File) Close() error {
	return file.file.close()
}

func (file *file) close() error {
	if file == nil || file.fd < 0 {
		return EINVAL
	}
	var e error
	if file.isdir() {
		e = syscall.FindClose(syscall.Handle(file.fd))
	} else {
		e = syscall.CloseHandle(syscall.Handle(file.fd))
	}
	var err error
	if e != nil {
		err = &PathError{"close", file.name, e}
	}
	file.fd = syscall.InvalidHandle // so it can't be closed again

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

// Readdir reads the contents of the directory associated with file and
// returns an array of up to n FileInfo structures, as would be returned
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
func (file *File) Readdir(n int) (fi []FileInfo, err error) {
	if file == nil || file.fd < 0 {
		return nil, EINVAL
	}
	if !file.isdir() {
		return nil, &PathError{"Readdir", file.name, ENOTDIR}
	}
	wantAll := n <= 0
	size := n
	if wantAll {
		n = -1
		size = 100
	}
	fi = make([]FileInfo, 0, size) // Empty with room to grow.
	d := &file.dirinfo.data
	for n != 0 {
		if file.dirinfo.needdata {
			e := syscall.FindNextFile(syscall.Handle(file.fd), d)
			if e != nil {
				if e == syscall.ERROR_NO_MORE_FILES {
					break
				} else {
					err = &PathError{"FindNextFile", file.name, e}
					if !wantAll {
						fi = nil
					}
					return
				}
			}
		}
		file.dirinfo.needdata = true
		name := string(syscall.UTF16ToString(d.FileName[0:]))
		if name == "." || name == ".." { // Useless names
			continue
		}
		f := toFileInfo(name, d.FileAttributes, d.FileSizeHigh, d.FileSizeLow, d.CreationTime, d.LastAccessTime, d.LastWriteTime)
		n--
		fi = append(fi, f)
	}
	if !wantAll && len(fi) == 0 {
		return fi, io.EOF
	}
	return fi, nil
}

// read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
func (f *File) read(b []byte) (n int, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	return syscall.Read(f.fd, b)
}

// pread reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to 0.
func (f *File) pread(b []byte, off int64) (n int, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	curoffset, e := syscall.Seek(f.fd, 0, 1)
	if e != nil {
		return 0, e
	}
	defer syscall.Seek(f.fd, curoffset, 0)
	o := syscall.Overlapped{
		OffsetHigh: uint32(off >> 32),
		Offset:     uint32(off),
	}
	var done uint32
	e = syscall.ReadFile(syscall.Handle(f.fd), b, &done, &o)
	if e != nil {
		return 0, e
	}
	return int(done), nil
}

// write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
func (f *File) write(b []byte) (n int, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	return syscall.Write(f.fd, b)
}

// pwrite writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
func (f *File) pwrite(b []byte, off int64) (n int, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	curoffset, e := syscall.Seek(f.fd, 0, 1)
	if e != nil {
		return 0, e
	}
	defer syscall.Seek(f.fd, curoffset, 0)
	o := syscall.Overlapped{
		OffsetHigh: uint32(off >> 32),
		Offset:     uint32(off),
	}
	var done uint32
	e = syscall.WriteFile(syscall.Handle(f.fd), b, &done, &o)
	if e != nil {
		return 0, e
	}
	return int(done), nil
}

// seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an error, if any.
func (f *File) seek(offset int64, whence int) (ret int64, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	return syscall.Seek(f.fd, offset, whence)
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
func Truncate(name string, size int64) error {
	f, e := OpenFile(name, O_WRONLY|O_CREATE, 0666)
	if e != nil {
		return e
	}
	defer f.Close()
	e1 := f.Truncate(size)
	if e1 != nil {
		return e1
	}
	return nil
}

// Pipe returns a connected pair of Files; reads from r return bytes written to w.
// It returns the files and an error, if any.
func Pipe() (r *File, w *File, err error) {
	var p [2]syscall.Handle

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
	const pathSep = '\\'
	dirw := make([]uint16, syscall.MAX_PATH)
	n, _ := syscall.GetTempPath(uint32(len(dirw)), &dirw[0])
	if n > uint32(len(dirw)) {
		dirw = make([]uint16, n)
		n, _ = syscall.GetTempPath(uint32(len(dirw)), &dirw[0])
		if n > uint32(len(dirw)) {
			n = 0
		}
	}
	if n > 0 && dirw[n-1] == pathSep {
		n--
	}
	return string(utf16.Decode(dirw[0:n]))
}
