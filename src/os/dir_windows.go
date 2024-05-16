// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/windows"
	"io"
	"io/fs"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

// Auxiliary information if the File describes a directory
type dirInfo struct {
	mu sync.Mutex
	// buf is a slice pointer so the slice header
	// does not escape to the heap when returning
	// buf to dirBufPool.
	buf   *[]byte // buffer for directory I/O
	bufp  int     // location of next record in buf
	h     syscall.Handle
	vol   uint32
	class uint32 // type of entries in buf
	path  string // absolute directory path, empty if the file system supports FILE_ID_BOTH_DIR_INFO
}

const (
	// dirBufSize is the size of the dirInfo buffer.
	// The buffer must be big enough to hold at least a single entry.
	// The filename alone can be 512 bytes (MAX_PATH*2), and the fixed part of
	// the FILE_ID_BOTH_DIR_INFO structure is 105 bytes, so dirBufSize
	// should not be set below 1024 bytes (512+105+safety buffer).
	// Windows 8.1 and earlier only works with buffer sizes up to 64 kB.
	dirBufSize = 64 * 1024 // 64kB
)

var dirBufPool = sync.Pool{
	New: func() any {
		// The buffer must be at least a block long.
		buf := make([]byte, dirBufSize)
		return &buf
	},
}

func (d *dirInfo) close() {
	d.h = 0
	if d.buf != nil {
		dirBufPool.Put(d.buf)
		d.buf = nil
	}
}

// allowReadDirFileID indicates whether File.readdir should try to use FILE_ID_BOTH_DIR_INFO
// if the underlying file system supports it.
// Useful for testing purposes.
var allowReadDirFileID = true

func (d *dirInfo) init(h syscall.Handle) {
	d.h = h
	d.class = windows.FileFullDirectoryRestartInfo
	// The previous settings are enough to read the directory entries.
	// The following code is only needed to support os.SameFile.

	// It is safe to query d.vol once and reuse the value.
	// Hard links are not allowed to reference files in other volumes.
	// Junctions and symbolic links can reference files and directories in other volumes,
	// but the reparse point should still live in the parent volume.
	var flags uint32
	err := windows.GetVolumeInformationByHandle(h, nil, 0, &d.vol, nil, &flags, nil, 0)
	if err != nil {
		d.vol = 0 // Set to zero in case Windows writes garbage to it.
		// If we can't get the volume information, we can't use os.SameFile,
		// but we can still read the directory entries.
		return
	}
	if flags&windows.FILE_SUPPORTS_OBJECT_IDS == 0 {
		// The file system does not support object IDs, no need to continue.
		return
	}
	if allowReadDirFileID && flags&windows.FILE_SUPPORTS_OPEN_BY_FILE_ID != 0 {
		// Use FileIdBothDirectoryRestartInfo if available as it returns the file ID
		// without the need to open the file.
		d.class = windows.FileIdBothDirectoryRestartInfo
	} else {
		// If FileIdBothDirectoryRestartInfo is not available but objects IDs are supported,
		// get the directory path so that os.SameFile can use it to open the file
		// and retrieve the file ID.
		d.path, _ = windows.FinalPath(h, windows.FILE_NAME_OPENED)
	}
}

func (file *File) readdir(n int, mode readdirMode) (names []string, dirents []DirEntry, infos []FileInfo, err error) {
	// If this file has no dirInfo, create one.
	var d *dirInfo
	for {
		d = file.dirinfo.Load()
		if d != nil {
			break
		}
		d = new(dirInfo)
		d.init(file.pfd.Sysfd)
		if file.dirinfo.CompareAndSwap(nil, d) {
			break
		}
		// We lost the race: try again.
		d.close()
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.buf == nil {
		d.buf = dirBufPool.Get().(*[]byte)
	}

	wantAll := n <= 0
	if wantAll {
		n = -1
	}
	for n != 0 {
		// Refill the buffer if necessary
		if d.bufp == 0 {
			err = windows.GetFileInformationByHandleEx(file.pfd.Sysfd, d.class, (*byte)(unsafe.Pointer(&(*d.buf)[0])), uint32(len(*d.buf)))
			runtime.KeepAlive(file)
			if err != nil {
				if err == syscall.ERROR_NO_MORE_FILES {
					// Optimization: we can return the buffer to the pool, there is nothing else to read.
					dirBufPool.Put(d.buf)
					d.buf = nil
					break
				}
				if err == syscall.ERROR_FILE_NOT_FOUND &&
					(d.class == windows.FileIdBothDirectoryRestartInfo || d.class == windows.FileFullDirectoryRestartInfo) {
					// GetFileInformationByHandleEx doesn't document the return error codes when the info class is FileIdBothDirectoryRestartInfo,
					// but MS-FSA 2.1.5.6.3 [1] specifies that the underlying file system driver should return STATUS_NO_SUCH_FILE when
					// reading an empty root directory, which is mapped to ERROR_FILE_NOT_FOUND by Windows.
					// Note that some file system drivers may never return this error code, as the spec allows to return the "." and ".."
					// entries in such cases, making the directory appear non-empty.
					// The chances of false positive are very low, as we know that the directory exists, else GetVolumeInformationByHandle
					// would have failed, and that the handle is still valid, as we haven't closed it.
					// See go.dev/issue/61159.
					// [1] https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-fsa/fa8194e0-53ec-413b-8315-e8fa85396fd8
					break
				}
				if s, _ := file.Stat(); s != nil && !s.IsDir() {
					err = &PathError{Op: "readdir", Path: file.name, Err: syscall.ENOTDIR}
				} else {
					err = &PathError{Op: "GetFileInformationByHandleEx", Path: file.name, Err: err}
				}
				return
			}
			if d.class == windows.FileIdBothDirectoryRestartInfo {
				d.class = windows.FileIdBothDirectoryInfo
			} else if d.class == windows.FileFullDirectoryRestartInfo {
				d.class = windows.FileFullDirectoryInfo
			}
		}
		// Drain the buffer
		var islast bool
		for n != 0 && !islast {
			var nextEntryOffset uint32
			var nameslice []uint16
			entry := unsafe.Pointer(&(*d.buf)[d.bufp])
			if d.class == windows.FileIdBothDirectoryInfo {
				info := (*windows.FILE_ID_BOTH_DIR_INFO)(entry)
				nextEntryOffset = info.NextEntryOffset
				nameslice = unsafe.Slice(&info.FileName[0], info.FileNameLength/2)
			} else {
				info := (*windows.FILE_FULL_DIR_INFO)(entry)
				nextEntryOffset = info.NextEntryOffset
				nameslice = unsafe.Slice(&info.FileName[0], info.FileNameLength/2)
			}
			d.bufp += int(nextEntryOffset)
			islast = nextEntryOffset == 0
			if islast {
				d.bufp = 0
			}
			if (len(nameslice) == 1 && nameslice[0] == '.') ||
				(len(nameslice) == 2 && nameslice[0] == '.' && nameslice[1] == '.') {
				// Ignore "." and ".." and avoid allocating a string for them.
				continue
			}
			name := syscall.UTF16ToString(nameslice)
			if mode == readdirName {
				names = append(names, name)
			} else {
				var f *fileStat
				if d.class == windows.FileIdBothDirectoryInfo {
					f = newFileStatFromFileIDBothDirInfo((*windows.FILE_ID_BOTH_DIR_INFO)(entry))
				} else {
					f = newFileStatFromFileFullDirInfo((*windows.FILE_FULL_DIR_INFO)(entry))
					if d.path != "" {
						// Defer appending the entry name to the parent directory path until
						// it is really needed, to avoid allocating a string that may not be used.
						// It is currently only used in os.SameFile.
						f.appendNameToPath = true
						f.path = d.path
					}
				}
				f.name = name
				f.vol = d.vol
				if mode == readdirDirEntry {
					dirents = append(dirents, dirEntry{f})
				} else {
					infos = append(infos, f)
				}
			}
			n--
		}
	}
	if !wantAll && len(names)+len(dirents)+len(infos) == 0 {
		return nil, nil, nil, io.EOF
	}
	return names, dirents, infos, nil
}

type dirEntry struct {
	fs *fileStat
}

func (de dirEntry) Name() string            { return de.fs.Name() }
func (de dirEntry) IsDir() bool             { return de.fs.IsDir() }
func (de dirEntry) Type() FileMode          { return de.fs.Mode().Type() }
func (de dirEntry) Info() (FileInfo, error) { return de.fs, nil }

func (de dirEntry) String() string {
	return fs.FormatDirEntry(de)
}
