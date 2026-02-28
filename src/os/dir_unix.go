// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || dragonfly || freebsd || (js && wasm) || linux || netbsd || openbsd || solaris

package os

import (
	"io"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

// Auxiliary information if the File describes a directory
type dirInfo struct {
	buf  *[]byte // buffer for directory I/O
	nbuf int     // length of buf; return value from Getdirentries
	bufp int     // location of next record in buf.
}

const (
	// More than 5760 to work around https://golang.org/issue/24015.
	blockSize = 8192
)

var dirBufPool = sync.Pool{
	New: func() any {
		// The buffer must be at least a block long.
		buf := make([]byte, blockSize)
		return &buf
	},
}

func (d *dirInfo) close() {
	if d.buf != nil {
		dirBufPool.Put(d.buf)
		d.buf = nil
	}
}

func (f *File) readdir(n int, mode readdirMode) (names []string, dirents []DirEntry, infos []FileInfo, err error) {
	// If this file has no dirinfo, create one.
	if f.dirinfo == nil {
		f.dirinfo = new(dirInfo)
		f.dirinfo.buf = dirBufPool.Get().(*[]byte)
	}
	d := f.dirinfo

	// Change the meaning of n for the implementation below.
	//
	// The n above was for the public interface of "if n <= 0,
	// Readdir returns all the FileInfo from the directory in a
	// single slice".
	//
	// But below, we use only negative to mean looping until the
	// end and positive to mean bounded, with positive
	// terminating at 0.
	if n == 0 {
		n = -1
	}

	for n != 0 {
		// Refill the buffer if necessary
		if d.bufp >= d.nbuf {
			d.bufp = 0
			var errno error
			d.nbuf, errno = f.pfd.ReadDirent(*d.buf)
			runtime.KeepAlive(f)
			if errno != nil {
				return names, dirents, infos, &PathError{Op: "readdirent", Path: f.name, Err: errno}
			}
			if d.nbuf <= 0 {
				break // EOF
			}
		}

		// Drain the buffer
		buf := (*d.buf)[d.bufp:d.nbuf]
		reclen, ok := direntReclen(buf)
		if !ok || reclen > uint64(len(buf)) {
			break
		}
		rec := buf[:reclen]
		d.bufp += int(reclen)
		ino, ok := direntIno(rec)
		if !ok {
			break
		}
		if ino == 0 {
			continue
		}
		const namoff = uint64(unsafe.Offsetof(syscall.Dirent{}.Name))
		namlen, ok := direntNamlen(rec)
		if !ok || namoff+namlen > uint64(len(rec)) {
			break
		}
		name := rec[namoff : namoff+namlen]
		for i, c := range name {
			if c == 0 {
				name = name[:i]
				break
			}
		}
		// Check for useless names before allocating a string.
		if string(name) == "." || string(name) == ".." {
			continue
		}
		if n > 0 { // see 'n == 0' comment above
			n--
		}
		if mode == readdirName {
			names = append(names, string(name))
		} else if mode == readdirDirEntry {
			de, err := newUnixDirent(f.name, string(name), direntType(rec))
			if IsNotExist(err) {
				// File disappeared between readdir and stat.
				// Treat as if it didn't exist.
				continue
			}
			if err != nil {
				return nil, dirents, nil, err
			}
			dirents = append(dirents, de)
		} else {
			info, err := lstat(f.name + "/" + string(name))
			if IsNotExist(err) {
				// File disappeared between readdir + stat.
				// Treat as if it didn't exist.
				continue
			}
			if err != nil {
				return nil, nil, infos, err
			}
			infos = append(infos, info)
		}
	}

	if n > 0 && len(names)+len(dirents)+len(infos) == 0 {
		return nil, nil, nil, io.EOF
	}
	return names, dirents, infos, nil
}

// readInt returns the size-bytes unsigned integer in native byte order at offset off.
func readInt(b []byte, off, size uintptr) (u uint64, ok bool) {
	if len(b) < int(off+size) {
		return 0, false
	}
	if isBigEndian {
		return readIntBE(b[off:], size), true
	}
	return readIntLE(b[off:], size), true
}

func readIntBE(b []byte, size uintptr) uint64 {
	switch size {
	case 1:
		return uint64(b[0])
	case 2:
		_ = b[1] // bounds check hint to compiler; see golang.org/issue/14808
		return uint64(b[1]) | uint64(b[0])<<8
	case 4:
		_ = b[3] // bounds check hint to compiler; see golang.org/issue/14808
		return uint64(b[3]) | uint64(b[2])<<8 | uint64(b[1])<<16 | uint64(b[0])<<24
	case 8:
		_ = b[7] // bounds check hint to compiler; see golang.org/issue/14808
		return uint64(b[7]) | uint64(b[6])<<8 | uint64(b[5])<<16 | uint64(b[4])<<24 |
			uint64(b[3])<<32 | uint64(b[2])<<40 | uint64(b[1])<<48 | uint64(b[0])<<56
	default:
		panic("syscall: readInt with unsupported size")
	}
}

func readIntLE(b []byte, size uintptr) uint64 {
	switch size {
	case 1:
		return uint64(b[0])
	case 2:
		_ = b[1] // bounds check hint to compiler; see golang.org/issue/14808
		return uint64(b[0]) | uint64(b[1])<<8
	case 4:
		_ = b[3] // bounds check hint to compiler; see golang.org/issue/14808
		return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24
	case 8:
		_ = b[7] // bounds check hint to compiler; see golang.org/issue/14808
		return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
			uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
	default:
		panic("syscall: readInt with unsupported size")
	}
}
