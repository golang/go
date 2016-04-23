// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"unsafe"
)

// Stat returns the FileInfo structure describing file.
// If there is an error, it will be of type *PathError.
func (file *File) Stat() (FileInfo, error) {
	if file == nil {
		return nil, ErrInvalid
	}
	if file == nil || file.fd < 0 {
		return nil, syscall.EINVAL
	}
	if file.isdir() {
		// I don't know any better way to do that for directory
		return Stat(file.dirinfo.path)
	}
	if file.name == DevNull {
		return &devNullStat, nil
	}

	ft, err := syscall.GetFileType(file.fd)
	if err != nil {
		return nil, &PathError{"GetFileType", file.name, err}
	}
	if ft == syscall.FILE_TYPE_PIPE {
		return &fileStat{name: basename(file.name), pipe: true}, nil
	}

	var d syscall.ByHandleFileInformation
	err = syscall.GetFileInformationByHandle(file.fd, &d)
	if err != nil {
		return nil, &PathError{"GetFileInformationByHandle", file.name, err}
	}
	return &fileStat{
		name: basename(file.name),
		sys: syscall.Win32FileAttributeData{
			FileAttributes: d.FileAttributes,
			CreationTime:   d.CreationTime,
			LastAccessTime: d.LastAccessTime,
			LastWriteTime:  d.LastWriteTime,
			FileSizeHigh:   d.FileSizeHigh,
			FileSizeLow:    d.FileSizeLow,
		},
		vol:   d.VolumeSerialNumber,
		idxhi: d.FileIndexHigh,
		idxlo: d.FileIndexLow,
		pipe:  false,
	}, nil
}

// Stat returns a FileInfo structure describing the named file.
// If there is an error, it will be of type *PathError.
func Stat(name string) (FileInfo, error) {
	var fi FileInfo
	var err error
	for {
		fi, err = Lstat(name)
		if err != nil {
			return fi, err
		}
		if fi.Mode()&ModeSymlink == 0 {
			return fi, nil
		}
		name, err = Readlink(name)
		if err != nil {
			return fi, err
		}
	}
}

// Lstat returns the FileInfo structure describing the named file.
// If the file is a symbolic link, the returned FileInfo
// describes the symbolic link. Lstat makes no attempt to follow the link.
// If there is an error, it will be of type *PathError.
func Lstat(name string) (FileInfo, error) {
	if len(name) == 0 {
		return nil, &PathError{"Lstat", name, syscall.Errno(syscall.ERROR_PATH_NOT_FOUND)}
	}
	if name == DevNull {
		return &devNullStat, nil
	}
	fs := &fileStat{name: basename(name)}
	namep, e := syscall.UTF16PtrFromString(name)
	if e != nil {
		return nil, &PathError{"Lstat", name, e}
	}
	e = syscall.GetFileAttributesEx(namep, syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&fs.sys)))
	if e != nil {
		return nil, &PathError{"GetFileAttributesEx", name, e}
	}
	fs.path = name
	if !isAbs(fs.path) {
		fs.path, e = syscall.FullPath(fs.path)
		if e != nil {
			return nil, e
		}
	}
	return fs, nil
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

func isAbs(path string) (b bool) {
	v := volumeName(path)
	if v == "" {
		return false
	}
	path = path[len(v):]
	if path == "" {
		return false
	}
	return IsPathSeparator(path[0])
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
	if l := len(path); l >= 5 && IsPathSeparator(path[0]) && IsPathSeparator(path[1]) &&
		!IsPathSeparator(path[2]) && path[2] != '.' {
		// first, leading `\\` and next shouldn't be `\`. its server name.
		for n := 3; n < l-1; n++ {
			// second, next '\' shouldn't be repeated.
			if IsPathSeparator(path[n]) {
				n++
				// third, following something characters. its share name.
				if !IsPathSeparator(path[n]) {
					if path[n] == '.' {
						break
					}
					for ; n < l; n++ {
						if IsPathSeparator(path[n]) {
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
