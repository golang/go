// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/poll"
	"internal/syscall/windows"
	"runtime"
	"syscall"
	"unicode/utf16"
	"unsafe"
)

// file is the real representation of *File.
// The extra level of indirection ensures that no clients of os
// can overwrite this data, which could cause the finalizer
// to close the wrong file descriptor.
type file struct {
	pfd     poll.FD
	name    string
	dirinfo *dirInfo // nil unless directory being read
}

// Fd returns the Windows handle referencing the open file.
// The handle is valid only until f.Close is called or f is garbage collected.
func (file *File) Fd() uintptr {
	if file == nil {
		return uintptr(syscall.InvalidHandle)
	}
	return uintptr(file.pfd.Sysfd)
}

// newFile returns a new File with the given file handle and name.
// Unlike NewFile, it does not check that h is syscall.InvalidHandle.
func newFile(h syscall.Handle, name string, kind string) *File {
	if kind == "file" {
		var m uint32
		if syscall.GetConsoleMode(h, &m) == nil {
			kind = "console"
		}
	}

	f := &File{&file{
		pfd: poll.FD{
			Sysfd:         h,
			IsStream:      true,
			ZeroReadIsEOF: true,
		},
		name: name,
	}}
	runtime.SetFinalizer(f.file, (*file).close)

	// Ignore initialization errors.
	// Assume any problems will show up in later I/O.
	f.pfd.Init(kind)

	return f
}

// newConsoleFile creates new File that will be used as console.
func newConsoleFile(h syscall.Handle, name string) *File {
	return newFile(h, name, "console")
}

// NewFile returns a new File with the given file descriptor and
// name. The returned value will be nil if fd is not a valid file
// descriptor.
func NewFile(fd uintptr, name string) *File {
	h := syscall.Handle(fd)
	if h == syscall.InvalidHandle {
		return nil
	}
	return newFile(h, name, "file")
}

// Auxiliary information if the File describes a directory
type dirInfo struct {
	data     syscall.Win32finddata
	needdata bool
	path     string
	isempty  bool // set if FindFirstFile returns ERROR_FILE_NOT_FOUND
}

func epipecheck(file *File, e error) {
}

const DevNull = "NUL"

func (f *file) isdir() bool { return f != nil && f.dirinfo != nil }

func openFile(name string, flag int, perm FileMode) (file *File, err error) {
	r, e := syscall.Open(fixLongPath(name), flag|syscall.O_CLOEXEC, syscallMode(perm))
	if e != nil {
		return nil, e
	}
	return newFile(r, name, "file"), nil
}

func openDir(name string) (file *File, err error) {
	var mask string

	path := fixLongPath(name)

	if len(path) == 2 && path[1] == ':' || (len(path) > 0 && path[len(path)-1] == '\\') { // it is a drive letter, like C:
		mask = path + `*`
	} else {
		mask = path + `\*`
	}
	maskp, e := syscall.UTF16PtrFromString(mask)
	if e != nil {
		return nil, e
	}
	d := new(dirInfo)
	r, e := syscall.FindFirstFile(maskp, &d.data)
	if e != nil {
		// FindFirstFile returns ERROR_FILE_NOT_FOUND when
		// no matching files can be found. Then, if directory
		// exists, we should proceed.
		if e != syscall.ERROR_FILE_NOT_FOUND {
			return nil, e
		}
		var fa syscall.Win32FileAttributeData
		pathp, e := syscall.UTF16PtrFromString(path)
		if e != nil {
			return nil, e
		}
		e = syscall.GetFileAttributesEx(pathp, syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&fa)))
		if e != nil {
			return nil, e
		}
		if fa.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY == 0 {
			return nil, e
		}
		d.isempty = true
	}
	d.path = path
	if !isAbs(d.path) {
		d.path, e = syscall.FullPath(d.path)
		if e != nil {
			return nil, e
		}
	}
	f := newFile(r, name, "dir")
	f.dirinfo = d
	return f, nil
}

// OpenFile is the generalized open call; most users will use Open
// or Create instead. It opens the named file with specified flag
// (O_RDONLY etc.) and perm, (0666 etc.) if applicable. If successful,
// methods on the returned File can be used for I/O.
// If there is an error, it will be of type *PathError.
func OpenFile(name string, flag int, perm FileMode) (*File, error) {
	if name == "" {
		return nil, &PathError{"open", name, syscall.ENOENT}
	}
	r, errf := openFile(name, flag, perm)
	if errf == nil {
		return r, nil
	}
	r, errd := openDir(name)
	if errd == nil {
		if flag&O_WRONLY != 0 || flag&O_RDWR != 0 {
			r.Close()
			return nil, &PathError{"open", name, syscall.EISDIR}
		}
		return r, nil
	}
	return nil, &PathError{"open", name, errf}
}

// Close closes the File, rendering it unusable for I/O.
// It returns an error, if any.
func (file *File) Close() error {
	if file == nil {
		return ErrInvalid
	}
	return file.file.close()
}

func (file *file) close() error {
	if file == nil {
		return syscall.EINVAL
	}
	if file.isdir() && file.dirinfo.isempty {
		// "special" empty directories
		return nil
	}
	var err error
	if e := file.pfd.Close(); e != nil {
		if e == poll.ErrFileClosing {
			e = ErrClosed
		}
		err = &PathError{"close", file.name, e}
	}

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

// read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
func (f *File) read(b []byte) (n int, err error) {
	n, err = f.pfd.Read(b)
	runtime.KeepAlive(f)
	return n, err
}

// pread reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to 0.
func (f *File) pread(b []byte, off int64) (n int, err error) {
	n, err = f.pfd.Pread(b, off)
	runtime.KeepAlive(f)
	return n, err
}

// write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
func (f *File) write(b []byte) (n int, err error) {
	n, err = f.pfd.Write(b)
	runtime.KeepAlive(f)
	return n, err
}

// pwrite writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
func (f *File) pwrite(b []byte, off int64) (n int, err error) {
	n, err = f.pfd.Pwrite(b, off)
	runtime.KeepAlive(f)
	return n, err
}

// seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an error, if any.
func (f *File) seek(offset int64, whence int) (ret int64, err error) {
	ret, err = f.pfd.Seek(offset, whence)
	runtime.KeepAlive(f)
	return ret, err
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

// Remove removes the named file or directory.
// If there is an error, it will be of type *PathError.
func Remove(name string) error {
	p, e := syscall.UTF16PtrFromString(fixLongPath(name))
	if e != nil {
		return &PathError{"remove", name, e}
	}

	// Go file interface forces us to know whether
	// name is a file or directory. Try both.
	e = syscall.DeleteFile(p)
	if e == nil {
		return nil
	}
	e1 := syscall.RemoveDirectory(p)
	if e1 == nil {
		return nil
	}

	// Both failed: figure out which error to return.
	if e1 != e {
		a, e2 := syscall.GetFileAttributes(p)
		if e2 != nil {
			e = e2
		} else {
			if a&syscall.FILE_ATTRIBUTE_DIRECTORY != 0 {
				e = e1
			} else if a&syscall.FILE_ATTRIBUTE_READONLY != 0 {
				if e1 = syscall.SetFileAttributes(p, a&^syscall.FILE_ATTRIBUTE_READONLY); e1 == nil {
					if e = syscall.DeleteFile(p); e == nil {
						return nil
					}
				}
			}
		}
	}
	return &PathError{"remove", name, e}
}

func rename(oldname, newname string) error {
	e := windows.Rename(fixLongPath(oldname), fixLongPath(newname))
	if e != nil {
		return &LinkError{"rename", oldname, newname, e}
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

	return newFile(p[0], "|0", "file"), newFile(p[1], "|1", "file"), nil
}

func tempDir() string {
	n := uint32(syscall.MAX_PATH)
	for {
		b := make([]uint16, n)
		n, _ = syscall.GetTempPath(uint32(len(b)), &b[0])
		if n > uint32(len(b)) {
			continue
		}
		if n > 0 && b[n-1] == '\\' {
			n--
		}
		return string(utf16.Decode(b[:n]))
	}
}

// Link creates newname as a hard link to the oldname file.
// If there is an error, it will be of type *LinkError.
func Link(oldname, newname string) error {
	n, err := syscall.UTF16PtrFromString(fixLongPath(newname))
	if err != nil {
		return &LinkError{"link", oldname, newname, err}
	}
	o, err := syscall.UTF16PtrFromString(fixLongPath(oldname))
	if err != nil {
		return &LinkError{"link", oldname, newname, err}
	}
	err = syscall.CreateHardLink(n, o, 0)
	if err != nil {
		return &LinkError{"link", oldname, newname, err}
	}
	return nil
}

// Symlink creates newname as a symbolic link to oldname.
// If there is an error, it will be of type *LinkError.
func Symlink(oldname, newname string) error {
	// CreateSymbolicLink is not supported before Windows Vista
	if syscall.LoadCreateSymbolicLink() != nil {
		return &LinkError{"symlink", oldname, newname, syscall.EWINDOWS}
	}

	// '/' does not work in link's content
	oldname = fromSlash(oldname)

	// need the exact location of the oldname when its relative to determine if its a directory
	destpath := oldname
	if !isAbs(oldname) {
		destpath = dirname(newname) + `\` + oldname
	}

	fi, err := Lstat(destpath)
	isdir := err == nil && fi.IsDir()

	n, err := syscall.UTF16PtrFromString(fixLongPath(newname))
	if err != nil {
		return &LinkError{"symlink", oldname, newname, err}
	}
	o, err := syscall.UTF16PtrFromString(fixLongPath(oldname))
	if err != nil {
		return &LinkError{"symlink", oldname, newname, err}
	}

	var flags uint32
	if isdir {
		flags |= syscall.SYMBOLIC_LINK_FLAG_DIRECTORY
	}
	err = syscall.CreateSymbolicLink(n, o, flags)
	if err != nil {
		return &LinkError{"symlink", oldname, newname, err}
	}
	return nil
}
