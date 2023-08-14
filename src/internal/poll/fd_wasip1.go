// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"sync/atomic"
	"syscall"
	"unsafe"
)

type SysFile struct {
	// RefCountPtr is a pointer to the reference count of Sysfd.
	//
	// WASI preview 1 lacks a dup(2) system call. When the os and net packages
	// need to share a file/socket, instead of duplicating the underlying file
	// descriptor, we instead provide a way to copy FD instances and manage the
	// underlying file descriptor with reference counting.
	RefCountPtr *int32

	// RefCount is the reference count of Sysfd. When a copy of an FD is made,
	// it points to the reference count of the original FD instance.
	RefCount int32

	// Cache for the file type, lazily initialized when Seek is called.
	Filetype uint32

	// If the file represents a directory, this field contains the current
	// readdir position. It is reset to zero if the program calls Seek(0, 0).
	Dircookie uint64

	// Absolute path of the file, as returned by syscall.PathOpen;
	// this is used by Fchdir to emulate setting the current directory
	// to an open file descriptor.
	Path string

	// TODO(achille): it could be meaningful to move isFile from FD to a method
	// on this struct type, and expose it as `IsFile() bool` which derives the
	// result from the Filetype field. We would need to ensure that Filetype is
	// always set instead of being lazily initialized.
}

func (s *SysFile) init() {
	if s.RefCountPtr == nil {
		s.RefCount = 1
		s.RefCountPtr = &s.RefCount
	}
}

func (s *SysFile) ref() SysFile {
	atomic.AddInt32(s.RefCountPtr, +1)
	return SysFile{RefCountPtr: s.RefCountPtr}
}

func (s *SysFile) destroy(fd int) error {
	if s.RefCountPtr != nil && atomic.AddInt32(s.RefCountPtr, -1) > 0 {
		return nil
	}

	// We don't use ignoringEINTR here because POSIX does not define
	// whether the descriptor is closed if close returns EINTR.
	// If the descriptor is indeed closed, using a loop would race
	// with some other goroutine opening a new descriptor.
	// (The Linux kernel guarantees that it is closed on an EINTR error.)
	return CloseFunc(fd)
}

// Copy creates a copy of the FD.
//
// The FD instance points to the same underlying file descriptor. The file
// descriptor isn't closed until all FD instances that refer to it have been
// closed/destroyed.
func (fd *FD) Copy() FD {
	return FD{
		Sysfd:         fd.Sysfd,
		SysFile:       fd.SysFile.ref(),
		IsStream:      fd.IsStream,
		ZeroReadIsEOF: fd.ZeroReadIsEOF,
		isBlocking:    fd.isBlocking,
		isFile:        fd.isFile,
	}
}

// dupCloseOnExecOld always errors on wasip1 because there is no mechanism to
// duplicate file descriptors.
func dupCloseOnExecOld(fd int) (int, string, error) {
	return -1, "dup", syscall.ENOSYS
}

// Fchdir wraps syscall.Fchdir.
func (fd *FD) Fchdir() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Chdir(fd.Path)
}

// ReadDir wraps syscall.ReadDir.
// We treat this like an ordinary system call rather than a call
// that tries to fill the buffer.
func (fd *FD) ReadDir(buf []byte, cookie syscall.Dircookie) (int, error) {
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()
	for {
		n, err := syscall.ReadDir(fd.Sysfd, buf, cookie)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		// Do not call eofError; caller does not expect to see io.EOF.
		return n, err
	}
}

func (fd *FD) ReadDirent(buf []byte) (int, error) {
	n, err := fd.ReadDir(buf, fd.Dircookie)
	if err != nil {
		return 0, err
	}
	if n <= 0 {
		return n, nil // EOF
	}

	// We assume that the caller of ReadDirent will consume the entire buffer
	// up to the last full entry, so we scan through the buffer looking for the
	// value of the last next cookie.
	b := buf[:n]

	for len(b) > 0 {
		next, ok := direntNext(b)
		if !ok {
			break
		}
		size, ok := direntReclen(b)
		if !ok {
			break
		}
		if size > uint64(len(b)) {
			break
		}
		fd.Dircookie = syscall.Dircookie(next)
		b = b[size:]
	}

	// Trim a potentially incomplete trailing entry; this is necessary because
	// the code in src/os/dir_unix.go does not deal well with partial values in
	// calls to direntReclen, etc... and ends up causing an early EOF before all
	// directory entries were consumed. ReadDirent is called with a large enough
	// buffer (8 KiB) that at least one entry should always fit, tho this seems
	// a bit brittle but cannot be addressed without a large change of the
	// algorithm in the os.(*File).readdir method.
	return n - len(b), nil
}

// Seek wraps syscall.Seek.
func (fd *FD) Seek(offset int64, whence int) (int64, error) {
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()
	// syscall.Filetype is a uint8 but we store it as a uint32 in SysFile in
	// order to use atomic load/store on the field, which is why we have to
	// perform this type conversion.
	fileType := syscall.Filetype(atomic.LoadUint32(&fd.Filetype))

	if fileType == syscall.FILETYPE_UNKNOWN {
		var stat syscall.Stat_t
		if err := fd.Fstat(&stat); err != nil {
			return 0, err
		}
		fileType = stat.Filetype
		atomic.StoreUint32(&fd.Filetype, uint32(fileType))
	}

	if fileType == syscall.FILETYPE_DIRECTORY {
		// If the file descriptor is opened on a directory, we reset the readdir
		// cookie when seeking back to the beginning to allow reusing the file
		// descriptor to scan the directory again.
		if offset == 0 && whence == 0 {
			fd.Dircookie = 0
			return 0, nil
		} else {
			return 0, syscall.EINVAL
		}
	}

	return syscall.Seek(fd.Sysfd, offset, whence)
}

// https://github.com/WebAssembly/WASI/blob/main/legacy/preview1/docs.md#-dirent-record
const sizeOfDirent = 24

func direntReclen(buf []byte) (uint64, bool) {
	namelen, ok := direntNamlen(buf)
	return sizeOfDirent + namelen, ok
}

func direntNamlen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(syscall.Dirent{}.Namlen), unsafe.Sizeof(syscall.Dirent{}.Namlen))
}

func direntNext(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(syscall.Dirent{}.Next), unsafe.Sizeof(syscall.Dirent{}.Next))
}

// readInt returns the size-bytes unsigned integer in native byte order at offset off.
func readInt(b []byte, off, size uintptr) (u uint64, ok bool) {
	if len(b) < int(off+size) {
		return 0, false
	}
	return readIntLE(b[off:], size), true
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
		panic("internal/poll: readInt with unsupported size")
	}
}
