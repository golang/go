// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
)

// Stat returns the FileInfo structure describing file.
// If there is an error, it will be of type *PathError.
func (file *File) Stat() (FileInfo, error) {
	if file == nil {
		return nil, ErrInvalid
	}

	if file.isdir() {
		// I don't know any better way to do that for directory
		return Stat(file.dirinfo.path)
	}
	if file.name == DevNull {
		return &devNullStat, nil
	}

	ft, err := file.pfd.GetFileType()
	if err != nil {
		return nil, &PathError{"GetFileType", file.name, err}
	}
	switch ft {
	case syscall.FILE_TYPE_PIPE, syscall.FILE_TYPE_CHAR:
		return &fileStat{name: basename(file.name), filetype: ft}, nil
	}

	fs, err := newFileStatFromGetFileInformationByHandle(file.name, file.pfd.Sysfd)
	if err != nil {
		return nil, err
	}
	fs.filetype = ft
	return fs, err
}

// statNolog implements Stat for Windows.
func statNolog(name string) (FileInfo, error) {
	if len(name) == 0 {
		return nil, &PathError{"Stat", name, syscall.Errno(syscall.ERROR_PATH_NOT_FOUND)}
	}
	if name == DevNull {
		return &devNullStat, nil
	}
	namep, err := syscall.UTF16PtrFromString(fixLongPath(name))
	if err != nil {
		return nil, &PathError{"Stat", name, err}
	}
	fs, err := newFileStatFromGetFileAttributesExOrFindFirstFile(name, namep)
	if err != nil {
		return nil, err
	}
	if !fs.isSymlink() {
		err = fs.updatePathAndName(name)
		if err != nil {
			return nil, err
		}
		return fs, nil
	}
	// Use Windows I/O manager to dereference the symbolic link, as per
	// https://blogs.msdn.microsoft.com/oldnewthing/20100212-00/?p=14963/
	h, err := syscall.CreateFile(namep, 0, 0, nil,
		syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return nil, &PathError{"CreateFile", name, err}
	}
	defer syscall.CloseHandle(h)

	return newFileStatFromGetFileInformationByHandle(name, h)
}

// lstatNolog implements Lstat for Windows.
func lstatNolog(name string) (FileInfo, error) {
	if len(name) == 0 {
		return nil, &PathError{"Lstat", name, syscall.Errno(syscall.ERROR_PATH_NOT_FOUND)}
	}
	if name == DevNull {
		return &devNullStat, nil
	}
	namep, err := syscall.UTF16PtrFromString(fixLongPath(name))
	if err != nil {
		return nil, &PathError{"Lstat", name, err}
	}
	fs, err := newFileStatFromGetFileAttributesExOrFindFirstFile(name, namep)
	if err != nil {
		return nil, err
	}
	err = fs.updatePathAndName(name)
	if err != nil {
		return nil, err
	}
	return fs, nil
}
