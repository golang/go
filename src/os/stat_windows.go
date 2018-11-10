// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/windows"
	"syscall"
	"unsafe"
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

	var d syscall.ByHandleFileInformation
	err = file.pfd.GetFileInformationByHandle(&d)
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
		filetype: ft,
		vol:      d.VolumeSerialNumber,
		idxhi:    d.FileIndexHigh,
		idxlo:    d.FileIndexLow,
	}, nil
}

// Stat returns a FileInfo structure describing the named file.
// If there is an error, it will be of type *PathError.
func Stat(name string) (FileInfo, error) {
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
	// Apparently (see https://github.com/golang/go/issues/19922#issuecomment-300031421)
	// GetFileAttributesEx is fastest approach to get file info.
	// It does not work for symlinks. But symlinks are rare,
	// so try GetFileAttributesEx first.
	var fs fileStat
	err = syscall.GetFileAttributesEx(namep, syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&fs.sys)))
	if err == nil && fs.sys.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT == 0 {
		fs.path = name
		if !isAbs(fs.path) {
			fs.path, err = syscall.FullPath(fs.path)
			if err != nil {
				return nil, &PathError{"FullPath", name, err}
			}
		}
		fs.name = basename(name)
		return &fs, nil
	}
	// Use Windows I/O manager to dereference the symbolic link, as per
	// https://blogs.msdn.microsoft.com/oldnewthing/20100212-00/?p=14963/
	h, err := syscall.CreateFile(namep, 0, 0, nil,
		syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		if err == windows.ERROR_SHARING_VIOLATION {
			// try FindFirstFile now that CreateFile failed
			return statWithFindFirstFile(name, namep)
		}
		return nil, &PathError{"CreateFile", name, err}
	}
	defer syscall.CloseHandle(h)

	var d syscall.ByHandleFileInformation
	err = syscall.GetFileInformationByHandle(h, &d)
	if err != nil {
		return nil, &PathError{"GetFileInformationByHandle", name, err}
	}
	return &fileStat{
		name: basename(name),
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
		// fileStat.path is used by os.SameFile to decide if it needs
		// to fetch vol, idxhi and idxlo. But these are already set,
		// so set fileStat.path to "" to prevent os.SameFile doing it again.
		// Also do not set fileStat.filetype, because it is only used for
		// console and stdin/stdout. But you cannot call os.Stat for these.
	}, nil
}

// statWithFindFirstFile is used by Stat to handle special case of statting
// c:\pagefile.sys. We might discover that other files need similar treatment.
func statWithFindFirstFile(name string, namep *uint16) (FileInfo, error) {
	var fd syscall.Win32finddata
	h, err := syscall.FindFirstFile(namep, &fd)
	if err != nil {
		return nil, &PathError{"FindFirstFile", name, err}
	}
	syscall.FindClose(h)

	fullpath := name
	if !isAbs(fullpath) {
		fullpath, err = syscall.FullPath(fullpath)
		if err != nil {
			return nil, &PathError{"FullPath", name, err}
		}
	}
	return &fileStat{
		name: basename(name),
		path: fullpath,
		sys: syscall.Win32FileAttributeData{
			FileAttributes: fd.FileAttributes,
			CreationTime:   fd.CreationTime,
			LastAccessTime: fd.LastAccessTime,
			LastWriteTime:  fd.LastWriteTime,
			FileSizeHigh:   fd.FileSizeHigh,
			FileSizeLow:    fd.FileSizeLow,
		},
	}, nil
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
	namep, e := syscall.UTF16PtrFromString(fixLongPath(name))
	if e != nil {
		return nil, &PathError{"Lstat", name, e}
	}
	e = syscall.GetFileAttributesEx(namep, syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&fs.sys)))
	if e != nil {
		if e != windows.ERROR_SHARING_VIOLATION {
			return nil, &PathError{"GetFileAttributesEx", name, e}
		}
		// try FindFirstFile now that GetFileAttributesEx failed
		return statWithFindFirstFile(name, namep)
	}
	fs.path = name
	if !isAbs(fs.path) {
		fs.path, e = syscall.FullPath(fs.path)
		if e != nil {
			return nil, &PathError{"FullPath", name, e}
		}
	}
	return fs, nil
}
