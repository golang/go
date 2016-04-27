// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"sync"
	"syscall"
	"time"
)

// A fileStat is the implementation of FileInfo returned by Stat and Lstat.
type fileStat struct {
	name string
	sys  syscall.Win32FileAttributeData
	pipe bool

	// used to implement SameFile
	sync.Mutex
	path  string
	vol   uint32
	idxhi uint32
	idxlo uint32
}

func (fs *fileStat) Size() int64 {
	return int64(fs.sys.FileSizeHigh)<<32 + int64(fs.sys.FileSizeLow)
}

func (fs *fileStat) Mode() (m FileMode) {
	if fs == &devNullStat {
		return ModeDevice | ModeCharDevice | 0666
	}
	if fs.sys.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY != 0 {
		m |= ModeDir | 0111
	}
	if fs.sys.FileAttributes&syscall.FILE_ATTRIBUTE_READONLY != 0 {
		m |= 0444
	} else {
		m |= 0666
	}
	if fs.sys.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		m |= ModeSymlink
	}
	if fs.pipe {
		m |= ModeNamedPipe
	}
	return m
}

func (fs *fileStat) ModTime() time.Time {
	return time.Unix(0, fs.sys.LastWriteTime.Nanoseconds())
}

// Sys returns syscall.Win32FileAttributeData for file fs.
func (fs *fileStat) Sys() interface{} { return &fs.sys }

func (fs *fileStat) loadFileId() error {
	fs.Lock()
	defer fs.Unlock()
	if fs.path == "" {
		// already done
		return nil
	}
	pathp, err := syscall.UTF16PtrFromString(fs.path)
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
