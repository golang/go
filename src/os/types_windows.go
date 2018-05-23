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

// newFileStatFromGetFileAttributesExOrFindFirstFile calls GetFileAttributesEx
// and FindFirstFile to gather all required information about the provided file path pathp.
func newFileStatFromGetFileAttributesExOrFindFirstFile(path string, pathp *uint16) (*fileStat, error) {
	// As suggested by Microsoft, use GetFileAttributes() to acquire the file information,
	// and if it's a reparse point use FindFirstFile() to get the tag:
	// https://msdn.microsoft.com/en-us/library/windows/desktop/aa363940(v=vs.85).aspx
	// Notice that always calling FindFirstFile can create performance problems
	// (https://golang.org/issues/19922#issuecomment-300031421)
	var fa syscall.Win32FileAttributeData
	err := syscall.GetFileAttributesEx(pathp, syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&fa)))
	if err == nil && fa.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT == 0 {
		// Not a symlink.
		return &fileStat{
			FileAttributes: fa.FileAttributes,
			CreationTime:   fa.CreationTime,
			LastAccessTime: fa.LastAccessTime,
			LastWriteTime:  fa.LastWriteTime,
			FileSizeHigh:   fa.FileSizeHigh,
			FileSizeLow:    fa.FileSizeLow,
		}, nil
	}
	// GetFileAttributesEx returns ERROR_INVALID_NAME if called
	// for invalid file name like "*.txt". Do not attempt to call
	// FindFirstFile with "*.txt", because FindFirstFile will
	// succeed. So just return ERROR_INVALID_NAME instead.
	// see https://golang.org/issue/24999 for details.
	if errno, _ := err.(syscall.Errno); errno == windows.ERROR_INVALID_NAME {
		return nil, &PathError{"GetFileAttributesEx", path, err}
	}
	// We might have symlink here. But some directories also have
	// FileAttributes FILE_ATTRIBUTE_REPARSE_POINT bit set.
	// For example, OneDrive directory is like that
	// (see golang.org/issue/22579 for details).
	// So use FindFirstFile instead to distinguish directories like
	// OneDrive from real symlinks (see instructions described at
	// https://blogs.msdn.microsoft.com/oldnewthing/20100212-00/?p=14963/
	// and in particular bits about using both FileAttributes and
	// Reserved0 fields).
	var fd syscall.Win32finddata
	sh, err := syscall.FindFirstFile(pathp, &fd)
	if err != nil {
		return nil, &PathError{"FindFirstFile", path, err}
	}
	syscall.FindClose(sh)

	return newFileStatFromWin32finddata(&fd), nil
}

func (fs *fileStat) updatePathAndName(name string) error {
	fs.path = name
	if !isAbs(fs.path) {
		var err error
		fs.path, err = syscall.FullPath(fs.path)
		if err != nil {
			return &PathError{"FullPath", name, err}
		}
	}
	fs.name = basename(name)
	return nil
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
	h, err := syscall.CreateFile(pathp, 0, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
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
