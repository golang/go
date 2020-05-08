// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/windows"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// A fileStat is the implementation of FileInfo returned by Stat and Lstat.
type fileStat struct {
	name string

	// from ByHandleFileInformation, Win32FileAttributeData and Win32finddata
	FileAttributes uint32
	CreationTime   syscall.Filetime
	LastAccessTime syscall.Filetime
	LastWriteTime  syscall.Filetime
	FileSizeHigh   uint32
	FileSizeLow    uint32

	// from Win32finddata
	Reserved0 uint32

	// what syscall.GetFileType returns
	filetype uint32

	// used to implement SameFile
	sync.Mutex
	path             string
	vol              uint32
	idxhi            uint32
	idxlo            uint32
	appendNameToPath bool
}

// newFileStatFromGetFileInformationByHandle calls GetFileInformationByHandle
// to gather all required information about the file handle h.
func newFileStatFromGetFileInformationByHandle(path string, h syscall.Handle) (fs *fileStat, err error) {
	var d syscall.ByHandleFileInformation
	err = syscall.GetFileInformationByHandle(h, &d)
	if err != nil {
		return nil, &PathError{"GetFileInformationByHandle", path, err}
	}

	var ti windows.FILE_ATTRIBUTE_TAG_INFO
	err = windows.GetFileInformationByHandleEx(h, windows.FileAttributeTagInfo, (*byte)(unsafe.Pointer(&ti)), uint32(unsafe.Sizeof(ti)))
	if err != nil {
		if errno, ok := err.(syscall.Errno); ok && errno == windows.ERROR_INVALID_PARAMETER {
			// It appears calling GetFileInformationByHandleEx with
			// FILE_ATTRIBUTE_TAG_INFO fails on FAT file system with
			// ERROR_INVALID_PARAMETER. Clear ti.ReparseTag in that
			// instance to indicate no symlinks are possible.
			ti.ReparseTag = 0
		} else {
			return nil, &PathError{"GetFileInformationByHandleEx", path, err}
		}
	}

	return &fileStat{
		name:           basename(path),
		FileAttributes: d.FileAttributes,
		CreationTime:   d.CreationTime,
		LastAccessTime: d.LastAccessTime,
		LastWriteTime:  d.LastWriteTime,
		FileSizeHigh:   d.FileSizeHigh,
		FileSizeLow:    d.FileSizeLow,
		vol:            d.VolumeSerialNumber,
		idxhi:          d.FileIndexHigh,
		idxlo:          d.FileIndexLow,
		Reserved0:      ti.ReparseTag,
		// fileStat.path is used by os.SameFile to decide if it needs
		// to fetch vol, idxhi and idxlo. But these are already set,
		// so set fileStat.path to "" to prevent os.SameFile doing it again.
	}, nil
}

// newFileStatFromWin32finddata copies all required information
// from syscall.Win32finddata d into the newly created fileStat.
func newFileStatFromWin32finddata(d *syscall.Win32finddata) *fileStat {
	return &fileStat{
		FileAttributes: d.FileAttributes,
		CreationTime:   d.CreationTime,
		LastAccessTime: d.LastAccessTime,
		LastWriteTime:  d.LastWriteTime,
		FileSizeHigh:   d.FileSizeHigh,
		FileSizeLow:    d.FileSizeLow,
		Reserved0:      d.Reserved0,
	}
}

func (fs *fileStat) isSymlink() bool {
	// Use instructions described at
	// https://blogs.msdn.microsoft.com/oldnewthing/20100212-00/?p=14963/
	// to recognize whether it's a symlink.
	if fs.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT == 0 {
		return false
	}
	return fs.Reserved0 == syscall.IO_REPARSE_TAG_SYMLINK ||
		fs.Reserved0 == windows.IO_REPARSE_TAG_MOUNT_POINT
}

func (fs *fileStat) Size() int64 {
	return int64(fs.FileSizeHigh)<<32 + int64(fs.FileSizeLow)
}

func (fs *fileStat) Mode() (m FileMode) {
	if fs == &devNullStat {
		return ModeDevice | ModeCharDevice | 0666
	}
	if fs.FileAttributes&syscall.FILE_ATTRIBUTE_READONLY != 0 {
		m |= 0444
	} else {
		m |= 0666
	}
	if fs.isSymlink() {
		return m | ModeSymlink
	}
	if fs.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY != 0 {
		m |= ModeDir | 0111
	}
	switch fs.filetype {
	case syscall.FILE_TYPE_PIPE:
		m |= ModeNamedPipe
	case syscall.FILE_TYPE_CHAR:
		m |= ModeDevice | ModeCharDevice
	}
	return m
}

func (fs *fileStat) ModTime() time.Time {
	return time.Unix(0, fs.LastWriteTime.Nanoseconds())
}

// Sys returns syscall.Win32FileAttributeData for file fs.
func (fs *fileStat) Sys() interface{} {
	return &syscall.Win32FileAttributeData{
		FileAttributes: fs.FileAttributes,
		CreationTime:   fs.CreationTime,
		LastAccessTime: fs.LastAccessTime,
		LastWriteTime:  fs.LastWriteTime,
		FileSizeHigh:   fs.FileSizeHigh,
		FileSizeLow:    fs.FileSizeLow,
	}
}

func (fs *fileStat) loadFileId() error {
	fs.Lock()
	defer fs.Unlock()
	if fs.path == "" {
		// already done
		return nil
	}
	var path string
	if fs.appendNameToPath {
		path = fs.path + `\` + fs.name
	} else {
		path = fs.path
	}
	pathp, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	attrs := uint32(syscall.FILE_FLAG_BACKUP_SEMANTICS)
	if fs.isSymlink() {
		// Use FILE_FLAG_OPEN_REPARSE_POINT, otherwise CreateFile will follow symlink.
		// See https://docs.microsoft.com/en-us/windows/desktop/FileIO/symbolic-link-effects-on-file-systems-functions#createfile-and-createfiletransacted
		attrs |= syscall.FILE_FLAG_OPEN_REPARSE_POINT
	}
	h, err := syscall.CreateFile(pathp, 0, 0, nil, syscall.OPEN_EXISTING, attrs, 0)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(h)
	var i syscall.ByHandleFileInformation
	err = syscall.GetFileInformationByHandle(h, &i)
	if err != nil {
		return err
	}
	fs.path = ""
	fs.vol = i.VolumeSerialNumber
	fs.idxhi = i.FileIndexHigh
	fs.idxlo = i.FileIndexLow
	return nil
}

// saveInfoFromPath saves full path of the file to be used by os.SameFile later,
// and set name from path.
func (fs *fileStat) saveInfoFromPath(path string) error {
	fs.path = path
	if !isAbs(fs.path) {
		var err error
		fs.path, err = syscall.FullPath(fs.path)
		if err != nil {
			return &PathError{"FullPath", path, err}
		}
	}
	fs.name = basename(path)
	return nil
}

// devNullStat is fileStat structure describing DevNull file ("NUL").
var devNullStat = fileStat{
	name: DevNull,
	// hopefully this will work for SameFile
	vol:   0,
	idxhi: 0,
	idxlo: 0,
}

func sameFile(fs1, fs2 *fileStat) bool {
	e := fs1.loadFileId()
	if e != nil {
		return false
	}
	e = fs2.loadFileId()
	if e != nil {
		return false
	}
	return fs1.vol == fs2.vol && fs1.idxhi == fs2.idxhi && fs1.idxlo == fs2.idxlo
}

// For testing.
func atime(fi FileInfo) time.Time {
	return time.Unix(0, fi.Sys().(*syscall.Win32FileAttributeData).LastAccessTime.Nanoseconds())
}
