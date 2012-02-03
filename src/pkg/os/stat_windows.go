// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"time"
	"unsafe"
)

// Stat returns the FileInfo structure describing file.
// It returns the FileInfo and an error, if any.
func (file *File) Stat() (fi FileInfo, err error) {
	if file == nil || file.fd < 0 {
		return nil, EINVAL
	}
	if file.isdir() {
		// I don't know any better way to do that for directory
		return Stat(file.name)
	}
	var d syscall.ByHandleFileInformation
	e := syscall.GetFileInformationByHandle(syscall.Handle(file.fd), &d)
	if e != nil {
		return nil, &PathError{"GetFileInformationByHandle", file.name, e}
	}
	return toFileInfo(basename(file.name), d.FileAttributes, d.FileSizeHigh, d.FileSizeLow, d.CreationTime, d.LastAccessTime, d.LastWriteTime), nil
}

// Stat returns a FileInfo structure describing the named file and an error, if any.
// If name names a valid symbolic link, the returned FileInfo describes
// the file pointed at by the link and has fi.FollowedSymlink set to true.
// If name names an invalid symbolic link, the returned FileInfo describes
// the link itself and has fi.FollowedSymlink set to false.
func Stat(name string) (fi FileInfo, err error) {
	if len(name) == 0 {
		return nil, &PathError{"Stat", name, syscall.Errno(syscall.ERROR_PATH_NOT_FOUND)}
	}
	var d syscall.Win32FileAttributeData
	e := syscall.GetFileAttributesEx(syscall.StringToUTF16Ptr(name), syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&d)))
	if e != nil {
		return nil, &PathError{"GetFileAttributesEx", name, e}
	}
	return toFileInfo(basename(name), d.FileAttributes, d.FileSizeHigh, d.FileSizeLow, d.CreationTime, d.LastAccessTime, d.LastWriteTime), nil
}

// Lstat returns the FileInfo structure describing the named file and an
// error, if any.  If the file is a symbolic link, the returned FileInfo
// describes the symbolic link.  Lstat makes no attempt to follow the link.
func Lstat(name string) (fi FileInfo, err error) {
	// No links on Windows
	return Stat(name)
}

// basename removes trailing slashes and the leading
// directory name and drive letter from path name.
func basename(name string) string {
	// Remove drive letter
	if len(name) == 2 && name[1] == ':' {
		name = "."
	} else if len(name) > 2 && name[1] == ':' {
		name = name[2:]
	}
	i := len(name) - 1
	// Remove trailing slashes
	for ; i > 0 && (name[i] == '/' || name[i] == '\\'); i-- {
		name = name[:i]
	}
	// Remove leading directory name
	for i--; i >= 0; i-- {
		if name[i] == '/' || name[i] == '\\' {
			name = name[i+1:]
			break
		}
	}
	return name
}

type winTimes struct {
	atime, ctime syscall.Filetime
}

func toFileInfo(name string, fa, sizehi, sizelo uint32, ctime, atime, mtime syscall.Filetime) FileInfo {
	fs := &fileStat{
		name:    name,
		size:    int64(sizehi)<<32 + int64(sizelo),
		modTime: time.Unix(0, mtime.Nanoseconds()),
		sys:     &winTimes{atime, ctime},
	}
	if fa&syscall.FILE_ATTRIBUTE_DIRECTORY != 0 {
		fs.mode |= ModeDir
	}
	if fa&syscall.FILE_ATTRIBUTE_READONLY != 0 {
		fs.mode |= 0444
	} else {
		fs.mode |= 0666
	}
	return fs
}

func sameFile(sys1, sys2 interface{}) bool {
	// TODO(rsc): Do better than this, but this matches what
	// used to happen when code compared .Dev and .Ino,
	// which were both always zero.  Obviously not all files
	// are the same.
	return true
}

// For testing.
func atime(fi FileInfo) time.Time {
	return time.Unix(0, fi.Sys().(*winTimes).atime.Nanoseconds())
}
