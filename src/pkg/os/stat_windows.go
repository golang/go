// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// Stat returns the FileInfo structure describing file.
// If there is an error, it will be of type *PathError.
func (file *File) Stat() (fi FileInfo, err error) {
	if file == nil || file.fd < 0 {
		return nil, syscall.EINVAL
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
	return &fileStat{
		name:    basename(file.name),
		size:    mkSize(d.FileSizeHigh, d.FileSizeLow),
		modTime: mkModTime(d.LastWriteTime),
		mode:    mkMode(d.FileAttributes),
		sys:     mkSysFromFI(&d),
	}, nil
}

// Stat returns a FileInfo structure describing the named file.
// If there is an error, it will be of type *PathError.
func Stat(name string) (fi FileInfo, err error) {
	if len(name) == 0 {
		return nil, &PathError{"Stat", name, syscall.Errno(syscall.ERROR_PATH_NOT_FOUND)}
	}
	var d syscall.Win32FileAttributeData
	e := syscall.GetFileAttributesEx(syscall.StringToUTF16Ptr(name), syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&d)))
	if e != nil {
		return nil, &PathError{"GetFileAttributesEx", name, e}
	}
	path := name
	if !isAbs(path) {
		cwd, _ := Getwd()
		path = cwd + `\` + path
	}
	return &fileStat{
		name:    basename(name),
		size:    mkSize(d.FileSizeHigh, d.FileSizeLow),
		modTime: mkModTime(d.LastWriteTime),
		mode:    mkMode(d.FileAttributes),
		sys:     mkSys(path, d.LastAccessTime, d.CreationTime),
	}, nil
}

// Lstat returns the FileInfo structure describing the named file.
// If the file is a symbolic link, the returned FileInfo
// describes the symbolic link.  Lstat makes no attempt to follow the link.
// If there is an error, it will be of type *PathError.
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

func isSlash(c uint8) bool {
	return c == '\\' || c == '/'
}

func isAbs(path string) (b bool) {
	v := volumeName(path)
	if v == "" {
		return false
	}
	path = path[len(v):]
	if path == "" {
		return false
	}
	return isSlash(path[0])
}

func volumeName(path string) (v string) {
	if len(path) < 2 {
		return ""
	}
	// with drive letter
	c := path[0]
	if path[1] == ':' &&
		('0' <= c && c <= '9' || 'a' <= c && c <= 'z' ||
			'A' <= c && c <= 'Z') {
		return path[:2]
	}
	// is it UNC
	if l := len(path); l >= 5 && isSlash(path[0]) && isSlash(path[1]) &&
		!isSlash(path[2]) && path[2] != '.' {
		// first, leading `\\` and next shouldn't be `\`. its server name.
		for n := 3; n < l-1; n++ {
			// second, next '\' shouldn't be repeated.
			if isSlash(path[n]) {
				n++
				// third, following something characters. its share name.
				if !isSlash(path[n]) {
					if path[n] == '.' {
						break
					}
					for ; n < l; n++ {
						if isSlash(path[n]) {
							break
						}
					}
					return path[:n]
				}
				break
			}
		}
	}
	return ""
}

type winSys struct {
	sync.Mutex
	path              string
	atime, ctime      syscall.Filetime
	vol, idxhi, idxlo uint32
}

func mkSize(hi, lo uint32) int64 {
	return int64(hi)<<32 + int64(lo)
}

func mkModTime(mtime syscall.Filetime) time.Time {
	return time.Unix(0, mtime.Nanoseconds())
}

func mkMode(fa uint32) (m FileMode) {
	if fa&syscall.FILE_ATTRIBUTE_DIRECTORY != 0 {
		m |= ModeDir
	}
	if fa&syscall.FILE_ATTRIBUTE_READONLY != 0 {
		m |= 0444
	} else {
		m |= 0666
	}
	return m
}

func mkSys(path string, atime, ctime syscall.Filetime) *winSys {
	return &winSys{
		path:  path,
		atime: atime,
		ctime: ctime,
	}
}

func mkSysFromFI(i *syscall.ByHandleFileInformation) *winSys {
	return &winSys{
		atime: i.LastAccessTime,
		ctime: i.CreationTime,
		vol:   i.VolumeSerialNumber,
		idxhi: i.FileIndexHigh,
		idxlo: i.FileIndexLow,
	}
}

func (s *winSys) loadFileId() error {
	if s.path == "" {
		// already done
		return nil
	}
	s.Lock()
	defer s.Unlock()
	h, e := syscall.CreateFile(syscall.StringToUTF16Ptr(s.path), syscall.GENERIC_READ, syscall.FILE_SHARE_READ, nil, syscall.OPEN_EXISTING, 0, 0)
	if e != nil {
		return e
	}
	defer syscall.CloseHandle(h)
	var i syscall.ByHandleFileInformation
	e = syscall.GetFileInformationByHandle(syscall.Handle(h), &i)
	if e != nil {
		return e
	}
	s.path = ""
	s.vol = i.VolumeSerialNumber
	s.idxhi = i.FileIndexHigh
	s.idxlo = i.FileIndexLow
	return nil
}

func sameFile(sys1, sys2 interface{}) bool {
	s1 := sys1.(*winSys)
	s2 := sys2.(*winSys)
	e := s1.loadFileId()
	if e != nil {
		panic(e)
	}
	e = s2.loadFileId()
	if e != nil {
		panic(e)
	}
	return s1.vol == s2.vol && s1.idxhi == s2.idxhi && s1.idxlo == s2.idxlo
}

// For testing.
func atime(fi FileInfo) time.Time {
	return time.Unix(0, fi.Sys().(*winSys).atime.Nanoseconds())
}
