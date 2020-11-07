// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd js,wasm linux netbsd openbsd solaris windows

package os

import (
	"runtime"
	"syscall"
	"time"
)

func sigpipe() // implemented in package runtime

// Close closes the File, rendering it unusable for I/O.
// On files that support SetDeadline, any pending I/O operations will
// be canceled and return immediately with an error.
// Close will return an error if it has already been called.
func (f *File) Close() error {
	if f == nil {
		return ErrInvalid
	}
	return f.file.close()
}

// read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
func (f *File) read(b []byte) (n int, err error) {
	n, err = f.pfd.Read(b)
	runtime.KeepAlive(f)
	return n, err
}

// pread reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to nil.
func (f *File) pread(b []byte, off int64) (n int, err error) {
	n, err = f.pfd.Pread(b, off)
	runtime.KeepAlive(f)
	return n, err
}

// write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
func (f *File) write(b []byte) (n int, err error) {
	n, err = f.pfd.Write(b)
	runtime.KeepAlive(f)
	return n, err
}

// pwrite writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
func (f *File) pwrite(b []byte, off int64) (n int, err error) {
	n, err = f.pfd.Pwrite(b, off)
	runtime.KeepAlive(f)
	return n, err
}

// syscallMode returns the syscall-specific mode bits from Go's portable mode bits.
func syscallMode(i FileMode) (o uint32) {
	o |= uint32(i.Perm())
	if i&ModeSetuid != 0 {
		o |= syscall.S_ISUID
	}
	if i&ModeSetgid != 0 {
		o |= syscall.S_ISGID
	}
	if i&ModeSticky != 0 {
		o |= syscall.S_ISVTX
	}
	// No mapping for Go's ModeTemporary (plan9 only).
	return
}

// See docs in file.go:Chmod.
func chmod(name string, mode FileMode) error {
	longName := fixLongPath(name)
	e := ignoringEINTR(func() error {
		return syscall.Chmod(longName, syscallMode(mode))
	})
	if e != nil {
		return &PathError{Op: "chmod", Path: name, Err: e}
	}
	return nil
}

// See docs in file.go:(*File).Chmod.
func (f *File) chmod(mode FileMode) error {
	if err := f.checkValid("chmod"); err != nil {
		return err
	}
	if e := f.pfd.Fchmod(syscallMode(mode)); e != nil {
		return f.wrapErr("chmod", e)
	}
	return nil
}

// Chown changes the numeric uid and gid of the named file.
// If the file is a symbolic link, it changes the uid and gid of the link's target.
// A uid or gid of -1 means to not change that value.
// If there is an error, it will be of type *PathError.
//
// On Windows or Plan 9, Chown always returns the syscall.EWINDOWS or
// EPLAN9 error, wrapped in *PathError.
func Chown(name string, uid, gid int) error {
	e := ignoringEINTR(func() error {
		return syscall.Chown(name, uid, gid)
	})
	if e != nil {
		return &PathError{Op: "chown", Path: name, Err: e}
	}
	return nil
}

// Lchown changes the numeric uid and gid of the named file.
// If the file is a symbolic link, it changes the uid and gid of the link itself.
// If there is an error, it will be of type *PathError.
//
// On Windows, it always returns the syscall.EWINDOWS error, wrapped
// in *PathError.
func Lchown(name string, uid, gid int) error {
	e := ignoringEINTR(func() error {
		return syscall.Lchown(name, uid, gid)
	})
	if e != nil {
		return &PathError{Op: "lchown", Path: name, Err: e}
	}
	return nil
}

// Chown changes the numeric uid and gid of the named file.
// If there is an error, it will be of type *PathError.
//
// On Windows, it always returns the syscall.EWINDOWS error, wrapped
// in *PathError.
func (f *File) Chown(uid, gid int) error {
	if err := f.checkValid("chown"); err != nil {
		return err
	}
	if e := f.pfd.Fchown(uid, gid); e != nil {
		return f.wrapErr("chown", e)
	}
	return nil
}

// Truncate changes the size of the file.
// It does not change the I/O offset.
// If there is an error, it will be of type *PathError.
func (f *File) Truncate(size int64) error {
	if err := f.checkValid("truncate"); err != nil {
		return err
	}
	if e := f.pfd.Ftruncate(size); e != nil {
		return f.wrapErr("truncate", e)
	}
	return nil
}

// Sync commits the current contents of the file to stable storage.
// Typically, this means flushing the file system's in-memory copy
// of recently written data to disk.
func (f *File) Sync() error {
	if err := f.checkValid("sync"); err != nil {
		return err
	}
	if e := f.pfd.Fsync(); e != nil {
		return f.wrapErr("sync", e)
	}
	return nil
}

// Chtimes changes the access and modification times of the named
// file, similar to the Unix utime() or utimes() functions.
//
// The underlying filesystem may truncate or round the values to a
// less precise time unit.
// If there is an error, it will be of type *PathError.
func Chtimes(name string, atime time.Time, mtime time.Time) error {
	var utimes [2]syscall.Timespec
	utimes[0] = syscall.NsecToTimespec(atime.UnixNano())
	utimes[1] = syscall.NsecToTimespec(mtime.UnixNano())
	if e := syscall.UtimesNano(fixLongPath(name), utimes[0:]); e != nil {
		return &PathError{Op: "chtimes", Path: name, Err: e}
	}
	return nil
}

// Chdir changes the current working directory to the file,
// which must be a directory.
// If there is an error, it will be of type *PathError.
func (f *File) Chdir() error {
	if err := f.checkValid("chdir"); err != nil {
		return err
	}
	if e := f.pfd.Fchdir(); e != nil {
		return f.wrapErr("chdir", e)
	}
	return nil
}

// setDeadline sets the read and write deadline.
func (f *File) setDeadline(t time.Time) error {
	if err := f.checkValid("SetDeadline"); err != nil {
		return err
	}
	return f.pfd.SetDeadline(t)
}

// setReadDeadline sets the read deadline.
func (f *File) setReadDeadline(t time.Time) error {
	if err := f.checkValid("SetReadDeadline"); err != nil {
		return err
	}
	return f.pfd.SetReadDeadline(t)
}

// setWriteDeadline sets the write deadline.
func (f *File) setWriteDeadline(t time.Time) error {
	if err := f.checkValid("SetWriteDeadline"); err != nil {
		return err
	}
	return f.pfd.SetWriteDeadline(t)
}

// checkValid checks whether f is valid for use.
// If not, it returns an appropriate error, perhaps incorporating the operation name op.
func (f *File) checkValid(op string) error {
	if f == nil {
		return ErrInvalid
	}
	return nil
}

// ignoringEINTR makes a function call and repeats it if it returns an
// EINTR error. This appears to be required even though we install all
// signal handlers with SA_RESTART: see #22838, #38033, #38836, #40846.
// Also #20400 and #36644 are issues in which a signal handler is
// installed without setting SA_RESTART. None of these are the common case,
// but there are enough of them that it seems that we can't avoid
// an EINTR loop.
func ignoringEINTR(fn func() error) error {
	for {
		err := fn()
		if err != syscall.EINTR {
			return err
		}
	}
}
