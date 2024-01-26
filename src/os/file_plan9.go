// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/bytealg"
	"internal/poll"
	"io"
	"runtime"
	"syscall"
	"time"
)

// fixLongPath is a noop on non-Windows platforms.
func fixLongPath(path string) string {
	return path
}

// file is the real representation of *File.
// The extra level of indirection ensures that no clients of os
// can overwrite this data, which could cause the finalizer
// to close the wrong file descriptor.
type file struct {
	fdmu       poll.FDMutex
	fd         int
	name       string
	dirinfo    *dirInfo // nil unless directory being read
	appendMode bool     // whether file is opened for appending
}

// Fd returns the integer Plan 9 file descriptor referencing the open file.
// If f is closed, the file descriptor becomes invalid.
// If f is garbage collected, a finalizer may close the file descriptor,
// making it invalid; see runtime.SetFinalizer for more information on when
// a finalizer might be run. On Unix systems this will cause the SetDeadline
// methods to stop working.
//
// As an alternative, see the f.SyscallConn method.
func (f *File) Fd() uintptr {
	if f == nil {
		return ^(uintptr(0))
	}
	return uintptr(f.fd)
}

// NewFile returns a new File with the given file descriptor and
// name. The returned value will be nil if fd is not a valid file
// descriptor.
func NewFile(fd uintptr, name string) *File {
	fdi := int(fd)
	if fdi < 0 {
		return nil
	}
	f := &File{&file{fd: fdi, name: name}}
	runtime.SetFinalizer(f.file, (*file).close)
	return f
}

// Auxiliary information if the File describes a directory
type dirInfo struct {
	buf  [syscall.STATMAX]byte // buffer for directory I/O
	nbuf int                   // length of buf; return value from Read
	bufp int                   // location of next record in buf.
}

func epipecheck(file *File, e error) {
}

// DevNull is the name of the operating system's “null device.”
// On Unix-like systems, it is "/dev/null"; on Windows, "NUL".
const DevNull = "/dev/null"

// syscallMode returns the syscall-specific mode bits from Go's portable mode bits.
func syscallMode(i FileMode) (o uint32) {
	o |= uint32(i.Perm())
	if i&ModeAppend != 0 {
		o |= syscall.DMAPPEND
	}
	if i&ModeExclusive != 0 {
		o |= syscall.DMEXCL
	}
	if i&ModeTemporary != 0 {
		o |= syscall.DMTMP
	}
	return
}

// openFileNolog is the Plan 9 implementation of OpenFile.
func openFileNolog(name string, flag int, perm FileMode) (*File, error) {
	var (
		fd     int
		e      error
		create bool
		excl   bool
		trunc  bool
		append bool
	)

	if flag&O_CREATE == O_CREATE {
		flag = flag & ^O_CREATE
		create = true
	}
	if flag&O_EXCL == O_EXCL {
		excl = true
	}
	if flag&O_TRUNC == O_TRUNC {
		trunc = true
	}
	// O_APPEND is emulated on Plan 9
	if flag&O_APPEND == O_APPEND {
		flag = flag &^ O_APPEND
		append = true
	}

	if (create && trunc) || excl {
		fd, e = syscall.Create(name, flag, syscallMode(perm))
	} else {
		fd, e = syscall.Open(name, flag)
		if IsNotExist(e) && create {
			fd, e = syscall.Create(name, flag, syscallMode(perm))
			if e != nil {
				return nil, &PathError{Op: "create", Path: name, Err: e}
			}
		}
	}

	if e != nil {
		return nil, &PathError{Op: "open", Path: name, Err: e}
	}

	if append {
		if _, e = syscall.Seek(fd, 0, io.SeekEnd); e != nil {
			return nil, &PathError{Op: "seek", Path: name, Err: e}
		}
	}

	return NewFile(uintptr(fd), name), nil
}

// Close closes the File, rendering it unusable for I/O.
// On files that support SetDeadline, any pending I/O operations will
// be canceled and return immediately with an ErrClosed error.
// Close will return an error if it has already been called.
func (f *File) Close() error {
	if f == nil {
		return ErrInvalid
	}
	return f.file.close()
}

func (file *file) close() error {
	if !file.fdmu.IncrefAndClose() {
		return &PathError{Op: "close", Path: file.name, Err: ErrClosed}
	}

	// At this point we should cancel any pending I/O.
	// How do we do that on Plan 9?

	err := file.decref()

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

// destroy actually closes the descriptor. This is called when
// there are no remaining references, by the decref, readUnlock,
// and writeUnlock methods.
func (file *file) destroy() error {
	var err error
	if e := syscall.Close(file.fd); e != nil {
		err = &PathError{Op: "close", Path: file.name, Err: e}
	}
	return err
}

// Stat returns the FileInfo structure describing file.
// If there is an error, it will be of type *PathError.
func (f *File) Stat() (FileInfo, error) {
	if f == nil {
		return nil, ErrInvalid
	}
	d, err := dirstat(f)
	if err != nil {
		return nil, err
	}
	return fileInfoFromStat(d), nil
}

// Truncate changes the size of the file.
// It does not change the I/O offset.
// If there is an error, it will be of type *PathError.
func (f *File) Truncate(size int64) error {
	if f == nil {
		return ErrInvalid
	}

	var d syscall.Dir
	d.Null()
	d.Length = size

	var buf [syscall.STATFIXLEN]byte
	n, err := d.Marshal(buf[:])
	if err != nil {
		return &PathError{Op: "truncate", Path: f.name, Err: err}
	}

	if err := f.incref("truncate"); err != nil {
		return err
	}
	defer f.decref()

	if err = syscall.Fwstat(f.fd, buf[:n]); err != nil {
		return &PathError{Op: "truncate", Path: f.name, Err: err}
	}
	return nil
}

const chmodMask = uint32(syscall.DMAPPEND | syscall.DMEXCL | syscall.DMTMP | ModePerm)

func (f *File) chmod(mode FileMode) error {
	if f == nil {
		return ErrInvalid
	}
	var d syscall.Dir

	odir, e := dirstat(f)
	if e != nil {
		return &PathError{Op: "chmod", Path: f.name, Err: e}
	}
	d.Null()
	d.Mode = odir.Mode&^chmodMask | syscallMode(mode)&chmodMask

	var buf [syscall.STATFIXLEN]byte
	n, err := d.Marshal(buf[:])
	if err != nil {
		return &PathError{Op: "chmod", Path: f.name, Err: err}
	}

	if err := f.incref("chmod"); err != nil {
		return err
	}
	defer f.decref()

	if err = syscall.Fwstat(f.fd, buf[:n]); err != nil {
		return &PathError{Op: "chmod", Path: f.name, Err: err}
	}
	return nil
}

// Sync commits the current contents of the file to stable storage.
// Typically, this means flushing the file system's in-memory copy
// of recently written data to disk.
func (f *File) Sync() error {
	if f == nil {
		return ErrInvalid
	}
	var d syscall.Dir
	d.Null()

	var buf [syscall.STATFIXLEN]byte
	n, err := d.Marshal(buf[:])
	if err != nil {
		return &PathError{Op: "sync", Path: f.name, Err: err}
	}

	if err := f.incref("sync"); err != nil {
		return err
	}
	defer f.decref()

	if err = syscall.Fwstat(f.fd, buf[:n]); err != nil {
		return &PathError{Op: "sync", Path: f.name, Err: err}
	}
	return nil
}

// read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
func (f *File) read(b []byte) (n int, err error) {
	if err := f.readLock(); err != nil {
		return 0, err
	}
	defer f.readUnlock()
	n, e := fixCount(syscall.Read(f.fd, b))
	if n == 0 && len(b) > 0 && e == nil {
		return 0, io.EOF
	}
	return n, e
}

// pread reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to nil.
func (f *File) pread(b []byte, off int64) (n int, err error) {
	if err := f.readLock(); err != nil {
		return 0, err
	}
	defer f.readUnlock()
	n, e := fixCount(syscall.Pread(f.fd, b, off))
	if n == 0 && len(b) > 0 && e == nil {
		return 0, io.EOF
	}
	return n, e
}

// write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
// Since Plan 9 preserves message boundaries, never allow
// a zero-byte write.
func (f *File) write(b []byte) (n int, err error) {
	if err := f.writeLock(); err != nil {
		return 0, err
	}
	defer f.writeUnlock()
	if len(b) == 0 {
		return 0, nil
	}
	return fixCount(syscall.Write(f.fd, b))
}

// pwrite writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
// Since Plan 9 preserves message boundaries, never allow
// a zero-byte write.
func (f *File) pwrite(b []byte, off int64) (n int, err error) {
	if err := f.writeLock(); err != nil {
		return 0, err
	}
	defer f.writeUnlock()
	if len(b) == 0 {
		return 0, nil
	}
	return fixCount(syscall.Pwrite(f.fd, b, off))
}

// seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an error, if any.
func (f *File) seek(offset int64, whence int) (ret int64, err error) {
	if err := f.incref(""); err != nil {
		return 0, err
	}
	defer f.decref()
	if f.dirinfo != nil {
		// Free cached dirinfo, so we allocate a new one if we
		// access this file as a directory again. See #35767 and #37161.
		f.dirinfo = nil
	}
	return syscall.Seek(f.fd, offset, whence)
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
// If there is an error, it will be of type *PathError.
func Truncate(name string, size int64) error {
	var d syscall.Dir

	d.Null()
	d.Length = size

	var buf [syscall.STATFIXLEN]byte
	n, err := d.Marshal(buf[:])
	if err != nil {
		return &PathError{Op: "truncate", Path: name, Err: err}
	}
	if err = syscall.Wstat(name, buf[:n]); err != nil {
		return &PathError{Op: "truncate", Path: name, Err: err}
	}
	return nil
}

// Remove removes the named file or directory.
// If there is an error, it will be of type *PathError.
func Remove(name string) error {
	if e := syscall.Remove(name); e != nil {
		return &PathError{Op: "remove", Path: name, Err: e}
	}
	return nil
}

// hasPrefix from the strings package.
func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[0:len(prefix)] == prefix
}

func rename(oldname, newname string) error {
	dirname := oldname[:bytealg.LastIndexByteString(oldname, '/')+1]
	if hasPrefix(newname, dirname) {
		newname = newname[len(dirname):]
	} else {
		return &LinkError{"rename", oldname, newname, ErrInvalid}
	}

	// If newname still contains slashes after removing the oldname
	// prefix, the rename is cross-directory and must be rejected.
	if bytealg.LastIndexByteString(newname, '/') >= 0 {
		return &LinkError{"rename", oldname, newname, ErrInvalid}
	}

	var d syscall.Dir

	d.Null()
	d.Name = newname

	buf := make([]byte, syscall.STATFIXLEN+len(d.Name))
	n, err := d.Marshal(buf[:])
	if err != nil {
		return &LinkError{"rename", oldname, newname, err}
	}

	// If newname already exists and is not a directory, rename replaces it.
	f, err := Stat(dirname + newname)
	if err == nil && !f.IsDir() {
		Remove(dirname + newname)
	}

	if err = syscall.Wstat(oldname, buf[:n]); err != nil {
		return &LinkError{"rename", oldname, newname, err}
	}
	return nil
}

// See docs in file.go:Chmod.
func chmod(name string, mode FileMode) error {
	var d syscall.Dir

	odir, e := dirstat(name)
	if e != nil {
		return &PathError{Op: "chmod", Path: name, Err: e}
	}
	d.Null()
	d.Mode = odir.Mode&^chmodMask | syscallMode(mode)&chmodMask

	var buf [syscall.STATFIXLEN]byte
	n, err := d.Marshal(buf[:])
	if err != nil {
		return &PathError{Op: "chmod", Path: name, Err: err}
	}
	if err = syscall.Wstat(name, buf[:n]); err != nil {
		return &PathError{Op: "chmod", Path: name, Err: err}
	}
	return nil
}

// Chtimes changes the access and modification times of the named
// file, similar to the Unix utime() or utimes() functions.
// A zero time.Time value will leave the corresponding file time unchanged.
//
// The underlying filesystem may truncate or round the values to a
// less precise time unit.
// If there is an error, it will be of type *PathError.
func Chtimes(name string, atime time.Time, mtime time.Time) error {
	var d syscall.Dir

	d.Null()
	d.Atime = uint32(atime.Unix())
	d.Mtime = uint32(mtime.Unix())
	if atime.IsZero() {
		d.Atime = 0xFFFFFFFF
	}
	if mtime.IsZero() {
		d.Mtime = 0xFFFFFFFF
	}

	var buf [syscall.STATFIXLEN]byte
	n, err := d.Marshal(buf[:])
	if err != nil {
		return &PathError{Op: "chtimes", Path: name, Err: err}
	}
	if err = syscall.Wstat(name, buf[:n]); err != nil {
		return &PathError{Op: "chtimes", Path: name, Err: err}
	}
	return nil
}

// Pipe returns a connected pair of Files; reads from r return bytes
// written to w. It returns the files and an error, if any.
func Pipe() (r *File, w *File, err error) {
	var p [2]int

	if e := syscall.Pipe(p[0:]); e != nil {
		return nil, nil, NewSyscallError("pipe", e)
	}

	return NewFile(uintptr(p[0]), "|0"), NewFile(uintptr(p[1]), "|1"), nil
}

// not supported on Plan 9

// Link creates newname as a hard link to the oldname file.
// If there is an error, it will be of type *LinkError.
func Link(oldname, newname string) error {
	return &LinkError{"link", oldname, newname, syscall.EPLAN9}
}

// Symlink creates newname as a symbolic link to oldname.
// On Windows, a symlink to a non-existent oldname creates a file symlink;
// if oldname is later created as a directory the symlink will not work.
// If there is an error, it will be of type *LinkError.
func Symlink(oldname, newname string) error {
	return &LinkError{"symlink", oldname, newname, syscall.EPLAN9}
}

func readlink(name string) (string, error) {
	return "", &PathError{Op: "readlink", Path: name, Err: syscall.EPLAN9}
}

// Chown changes the numeric uid and gid of the named file.
// If the file is a symbolic link, it changes the uid and gid of the link's target.
// A uid or gid of -1 means to not change that value.
// If there is an error, it will be of type *PathError.
//
// On Windows or Plan 9, Chown always returns the syscall.EWINDOWS or
// EPLAN9 error, wrapped in *PathError.
func Chown(name string, uid, gid int) error {
	return &PathError{Op: "chown", Path: name, Err: syscall.EPLAN9}
}

// Lchown changes the numeric uid and gid of the named file.
// If the file is a symbolic link, it changes the uid and gid of the link itself.
// If there is an error, it will be of type *PathError.
func Lchown(name string, uid, gid int) error {
	return &PathError{Op: "lchown", Path: name, Err: syscall.EPLAN9}
}

// Chown changes the numeric uid and gid of the named file.
// If there is an error, it will be of type *PathError.
func (f *File) Chown(uid, gid int) error {
	if f == nil {
		return ErrInvalid
	}
	return &PathError{Op: "chown", Path: f.name, Err: syscall.EPLAN9}
}

func tempDir() string {
	dir := Getenv("TMPDIR")
	if dir == "" {
		dir = "/tmp"
	}
	return dir
}

// Chdir changes the current working directory to the file,
// which must be a directory.
// If there is an error, it will be of type *PathError.
func (f *File) Chdir() error {
	if err := f.incref("chdir"); err != nil {
		return err
	}
	defer f.decref()
	if e := syscall.Fchdir(f.fd); e != nil {
		return &PathError{Op: "chdir", Path: f.name, Err: e}
	}
	return nil
}

// setDeadline sets the read and write deadline.
func (f *File) setDeadline(time.Time) error {
	if err := f.checkValid("SetDeadline"); err != nil {
		return err
	}
	return poll.ErrNoDeadline
}

// setReadDeadline sets the read deadline.
func (f *File) setReadDeadline(time.Time) error {
	if err := f.checkValid("SetReadDeadline"); err != nil {
		return err
	}
	return poll.ErrNoDeadline
}

// setWriteDeadline sets the write deadline.
func (f *File) setWriteDeadline(time.Time) error {
	if err := f.checkValid("SetWriteDeadline"); err != nil {
		return err
	}
	return poll.ErrNoDeadline
}

// checkValid checks whether f is valid for use, but does not prepare
// to actually use it. If f is not ready checkValid returns an appropriate
// error, perhaps incorporating the operation name op.
func (f *File) checkValid(op string) error {
	if f == nil {
		return ErrInvalid
	}
	if err := f.incref(op); err != nil {
		return err
	}
	return f.decref()
}

type rawConn struct{}

func (c *rawConn) Control(f func(uintptr)) error {
	return syscall.EPLAN9
}

func (c *rawConn) Read(f func(uintptr) bool) error {
	return syscall.EPLAN9
}

func (c *rawConn) Write(f func(uintptr) bool) error {
	return syscall.EPLAN9
}

func newRawConn(file *File) (*rawConn, error) {
	return nil, syscall.EPLAN9
}

func ignoringEINTR(fn func() error) error {
	return fn()
}
