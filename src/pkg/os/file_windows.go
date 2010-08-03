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
	stat         syscall.Stat_t
	usefirststat bool
}

const DevNull = "NUL"

func (file *File) isdir() bool { return file != nil && file.dirinfo != nil }

func openFile(name string, flag int, perm uint32) (file *File, err Error) {
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

func openDir(name string) (file *File, err Error) {
	d := new(dirInfo)
	r, e := syscall.FindFirstFile(syscall.StringToUTF16Ptr(name+"\\*"), &d.stat.Windata)
	if e != 0 {
		return nil, &PathError{"open", name, Errno(e)}
	}
	f := NewFile(int(r), name)
	d.usefirststat = true
	f.dirinfo = d
	return f, nil
}

// Open opens the named file with specified flag (O_RDONLY etc.) and perm, (0666 etc.)
// if applicable.  If successful, methods on the returned File can be used for I/O.
// It returns the File and an Error, if any.
func Open(name string, flag int, perm uint32) (file *File, err Error) {
	// TODO(brainman): not sure about my logic of assuming it is dir first, then fall back to file
	r, e := openDir(name)
	if e == nil {
		return r, nil
	}
	r, e = openFile(name, flag, perm)
	if e == nil {
		return r, nil
	}
	return nil, e
}

// Close closes the File, rendering it unusable for I/O.
// It returns an Error, if any.
func (file *File) Close() Error {
	if file == nil || file.fd < 0 {
		return EINVAL
	}
	var e int
	if file.isdir() {
		_, e = syscall.FindClose(int32(file.fd))
	} else {
		_, e = syscall.CloseHandle(int32(file.fd))
	}
	var err Error
	if e != 0 {
		err = &PathError{"close", file.name, Errno(e)}
	}
	file.fd = -1 // so it can't be closed again

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

func (file *File) statFile(name string) (fi *FileInfo, err Error) {
	var stat syscall.ByHandleFileInformation
	if ok, e := syscall.GetFileInformationByHandle(int32(file.fd), &stat); !ok {
		return nil, &PathError{"stat", file.name, Errno(e)}
	}
	return fileInfoFromByHandleInfo(new(FileInfo), file.name, &stat), nil
}

// Stat returns the FileInfo structure describing file.
// It returns the FileInfo and an error, if any.
func (file *File) Stat() (fi *FileInfo, err Error) {
	if file == nil || file.fd < 0 {
		return nil, EINVAL
	}
	if file.isdir() {
		// I don't know any better way to do that for directory
		return Stat(file.name)
	}
	return file.statFile(file.name)
}

// Readdir reads the contents of the directory associated with file and
// returns an array of up to count FileInfo structures, as would be returned
// by Stat, in directory order.  Subsequent calls on the same file will yield
// further FileInfos.
// A negative count means to read until EOF.
// Readdir returns the array and an Error, if any.
func (file *File) Readdir(count int) (fi []FileInfo, err Error) {
	di := file.dirinfo
	size := count
	if size < 0 {
		size = 100
	}
	fi = make([]FileInfo, 0, size) // Empty with room to grow.
	for count != 0 {
		if di.usefirststat {
			di.usefirststat = false
		} else {
			_, e := syscall.FindNextFile(int32(file.fd), &di.stat.Windata)
			if e != 0 {
				if e == syscall.ERROR_NO_MORE_FILES {
					break
				} else {
					return nil, &PathError{"FindNextFile", file.name, Errno(e)}
				}
			}
		}
		var f FileInfo
		fileInfoFromWin32finddata(&f, &di.stat.Windata)
		if f.Name == "." || f.Name == ".." { // Useless names
			continue
		}
		count--
		if len(fi) == cap(fi) {
			nfi := make([]FileInfo, len(fi), 2*len(fi))
			for i := 0; i < len(fi); i++ {
				nfi[i] = fi[i]
			}
			fi = nfi
		}
		fi = fi[0 : len(fi)+1]
		fi[len(fi)-1] = f
	}
	return fi, nil
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
func Truncate(name string, size int64) Error {
	f, e := Open(name, O_WRONLY|O_CREAT, 0666)
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
