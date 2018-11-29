// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/windows"
	"syscall"
	"unsafe"
)

// isNulName reports whether name is NUL file name.
// For example, it returns true for both "NUL" and "nul".
func isNulName(name string) bool {
	if len(name) != 3 {
		return false
	}
	if name[0] != 'n' && name[0] != 'N' {
		return false
	}
	if name[1] != 'u' && name[1] != 'U' {
		return false
	}
	if name[2] != 'l' && name[2] != 'L' {
		return false
	}
	return true
}

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
	if isNulName(file.name) {
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

// stat implements both Stat and Lstat of a file.
func stat(funcname, name string, createFileAttrs uint32) (FileInfo, error) {
	if len(name) == 0 {
		return nil, &PathError{funcname, name, syscall.Errno(syscall.ERROR_PATH_NOT_FOUND)}
	}
	if isNulName(name) {
		return &devNullStat, nil
	}
	namep, err := syscall.UTF16PtrFromString(fixLongPath(name))
	if err != nil {
		return nil, &PathError{funcname, name, err}
	}

	// Try GetFileAttributesEx first, because it is faster than CreateFile.
	// See https://golang.org/issues/19922#issuecomment-300031421 for details.
	var fa syscall.Win32FileAttributeData
	err = syscall.GetFileAttributesEx(namep, syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&fa)))
	if err == nil && fa.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT == 0 {
		// Not a symlink.
		fs := &fileStat{
			path:           name,
			FileAttributes: fa.FileAttributes,
			CreationTime:   fa.CreationTime,
			LastAccessTime: fa.LastAccessTime,
			LastWriteTime:  fa.LastWriteTime,
			FileSizeHigh:   fa.FileSizeHigh,
			FileSizeLow:    fa.FileSizeLow,
		}
		// Gather full path to be used by os.SameFile later.
		if !isAbs(fs.path) {
			fs.path, err = syscall.FullPath(fs.path)
			if err != nil {
				return nil, &PathError{"FullPath", name, err}
			}
		}
		fs.name = basename(name)
		return fs, nil
	}
	// GetFileAttributesEx fails with ERROR_SHARING_VIOLATION error for
	// files, like c:\pagefile.sys. Use FindFirstFile for such files.
	if err == windows.ERROR_SHARING_VIOLATION {
		var fd syscall.Win32finddata
		sh, err := syscall.FindFirstFile(namep, &fd)
		if err != nil {
			return nil, &PathError{"FindFirstFile", name, err}
		}
		syscall.FindClose(sh)
		return newFileStatFromWin32finddata(&fd), nil
	}

	// Finally use CreateFile.
	h, err := syscall.CreateFile(namep, 0, 0, nil,
		syscall.OPEN_EXISTING, createFileAttrs, 0)
	if err != nil {
		return nil, &PathError{"CreateFile", name, err}
	}
	defer syscall.CloseHandle(h)

	return newFileStatFromGetFileInformationByHandle(name, h)
}

// statNolog implements Stat for Windows.
func statNolog(name string) (FileInfo, error) {
	return stat("Stat", name, syscall.FILE_FLAG_BACKUP_SEMANTICS)
}

// lstatNolog implements Lstat for Windows.
func lstatNolog(name string) (FileInfo, error) {
	attrs := uint32(syscall.FILE_FLAG_BACKUP_SEMANTICS)
	// Use FILE_FLAG_OPEN_REPARSE_POINT, otherwise CreateFile will follow symlink.
	// See https://docs.microsoft.com/en-us/windows/desktop/FileIO/symbolic-link-effects-on-file-systems-functions#createfile-and-createfiletransacted
	attrs |= syscall.FILE_FLAG_OPEN_REPARSE_POINT
	return stat("Lstat", name, attrs)
}
