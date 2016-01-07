// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package os

import (
	"runtime"
	"syscall"
)

func sameFile(fs1, fs2 *fileStat) bool {
	return fs1.sys.Dev == fs2.sys.Dev && fs1.sys.Ino == fs2.sys.Ino
}

func rename(oldname, newname string) error {
	e := syscall.Rename(oldname, newname)
	if e != nil {
		return &LinkError{"rename", oldname, newname, e}
	}
	return nil
}

// File represents an open file descriptor.
type File struct {
	*file
}

// file is the real representation of *File.
// The extra level of indirection ensures that no clients of os
// can overwrite this data, which could cause the finalizer
// to close the wrong file descriptor.
type file struct {
	fd      int
	name    string
	dirinfo *dirInfo // nil unless directory being read
}

// Fd returns the integer Unix file descriptor referencing the open file.
// The file descriptor is valid only until f.Close is called or f is garbage collected.
func (f *File) Fd() uintptr {
	if f == nil {
		return ^(uintptr(0))
	}
	return uintptr(f.fd)
}

// NewFile returns a new File with the given file descriptor and name.
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
	buf  []byte // buffer for directory I/O
	nbuf int    // length of buf; return value from Getdirentries
	bufp int    // location of next record in buf.
}

// epipecheck raises SIGPIPE if we get an EPIPE error on standard
// output or standard error. See the SIGPIPE docs in os/signal, and
// issue 11845.
func epipecheck(file *File, e error) {
	if e == syscall.EPIPE && (file.fd == 1 || file.fd == 2) {
		sigpipe()
	}
}

// DevNull is the name of the operating system's ``null device.''
// On Unix-like systems, it is "/dev/null"; on Windows, "NUL".
const DevNull = "/dev/null"

// OpenFile is the generalized open call; most users will use Open
// or Create instead.  It opens the named file with specified flag
// (O_RDONLY etc.) and perm, (0666 etc.) if applicable.  If successful,
// methods on the returned File can be used for I/O.
// If there is an error, it will be of type *PathError.
func OpenFile(name string, flag int, perm FileMode) (*File, error) {
	chmod := false
	if !supportsCreateWithStickyBit && flag&O_CREATE != 0 && perm&ModeSticky != 0 {
		if _, err := Stat(name); IsNotExist(err) {
			chmod = true
		}
	}

	var r int
	for {
		var e error
		r, e = syscall.Open(name, flag|syscall.O_CLOEXEC, syscallMode(perm))
		if e == nil {
			break
		}

		// On OS X, sigaction(2) doesn't guarantee that SA_RESTART will cause
		// open(2) to be restarted for regular files. This is easy to reproduce on
		// fuse file systems (see http://golang.org/issue/11180).
		if runtime.GOOS == "darwin" && e == syscall.EINTR {
			continue
		}

		return nil, &PathError{"open", name, e}
	}

	// open(2) itself won't handle the sticky bit on *BSD and Solaris
	if chmod {
		Chmod(name, perm)
	}

	// There's a race here with fork/exec, which we are
	// content to live with.  See ../syscall/exec_unix.go.
	if !supportsCloseOnExec {
		syscall.CloseOnExec(r)
	}

	return NewFile(uintptr(r), name), nil
}

// Close closes the File, rendering it unusable for I/O.
// It returns an error, if any.
func (f *File) Close() error {
	if f == nil {
		return ErrInvalid
	}
	return f.file.close()
}

func (file *file) close() error {
	if file == nil || file.fd < 0 {
		return syscall.EINVAL
	}
	var err error
	if e := syscall.Close(file.fd); e != nil {
		err = &PathError{"close", file.name, e}
	}
	file.fd = -1 // so it can't be closed again

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

// Stat returns the FileInfo structure describing file.
// If there is an error, it will be of type *PathError.
func (f *File) Stat() (FileInfo, error) {
	if f == nil {
		return nil, ErrInvalid
	}
	var fs fileStat
	err := syscall.Fstat(f.fd, &fs.sys)
	if err != nil {
		return nil, &PathError{"stat", f.name, err}
	}
	fillFileStatFromSys(&fs, f.name)
	return &fs, nil
}

// Stat returns a FileInfo describing the named file.
// If there is an error, it will be of type *PathError.
func Stat(name string) (FileInfo, error) {
	var fs fileStat
	err := syscall.Stat(name, &fs.sys)
	if err != nil {
		return nil, &PathError{"stat", name, err}
	}
	fillFileStatFromSys(&fs, name)
	return &fs, nil
}

// Lstat returns a FileInfo describing the named file.
// If the file is a symbolic link, the returned FileInfo
// describes the symbolic link.  Lstat makes no attempt to follow the link.
// If there is an error, it will be of type *PathError.
func Lstat(name string) (FileInfo, error) {
	var fs fileStat
	err := syscall.Lstat(name, &fs.sys)
	if err != nil {
		return nil, &PathError{"lstat", name, err}
	}
	fillFileStatFromSys(&fs, name)
	return &fs, nil
}

func (f *File) readdir(n int) (fi []FileInfo, err error) {
	dirname := f.name
	if dirname == "" {
		dirname = "."
	}
	names, err := f.Readdirnames(n)
	fi = make([]FileInfo, 0, len(names))
	for _, filename := range names {
		fip, lerr := lstat(dirname + "/" + filename)
		if IsNotExist(lerr) {
			// File disappeared between readdir + stat.
			// Just treat it as if it didn't exist.
			continue
		}
		if lerr != nil {
			return fi, lerr
		}
		fi = append(fi, fip)
	}
	return fi, err
}

// Darwin and FreeBSD can't read or write 2GB+ at a time,
// even on 64-bit systems. See golang.org/issue/7812.
// Use 1GB instead of, say, 2GB-1, to keep subsequent
// reads aligned.
const (
	needsMaxRW = runtime.GOOS == "darwin" || runtime.GOOS == "freebsd"
	maxRW      = 1 << 30
)

// read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
func (f *File) read(b []byte) (n int, err error) {
	if needsMaxRW && len(b) > maxRW {
		b = b[:maxRW]
	}
	return fixCount(syscall.Read(f.fd, b))
}

// pread reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to nil.
func (f *File) pread(b []byte, off int64) (n int, err error) {
	if needsMaxRW && len(b) > maxRW {
		b = b[:maxRW]
	}
	return fixCount(syscall.Pread(f.fd, b, off))
}

// write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
func (f *File) write(b []byte) (n int, err error) {
	for {
		bcap := b
		if needsMaxRW && len(bcap) > maxRW {
			bcap = bcap[:maxRW]
		}
		m, err := fixCount(syscall.Write(f.fd, bcap))
		n += m

		// If the syscall wrote some data but not all (short write)
		// or it returned EINTR, then assume it stopped early for
		// reasons that are uninteresting to the caller, and try again.
		if 0 < m && m < len(bcap) || err == syscall.EINTR {
			b = b[m:]
			continue
		}

		if needsMaxRW && len(bcap) != len(b) && err == nil {
			b = b[m:]
			continue
		}

		return n, err
	}
}

// pwrite writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
func (f *File) pwrite(b []byte, off int64) (n int, err error) {
	if needsMaxRW && len(b) > maxRW {
		b = b[:maxRW]
	}
	return fixCount(syscall.Pwrite(f.fd, b, off))
}

// seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an error, if any.
func (f *File) seek(offset int64, whence int) (ret int64, err error) {
	return syscall.Seek(f.fd, offset, whence)
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
// If there is an error, it will be of type *PathError.
func Truncate(name string, size int64) error {
	if e := syscall.Truncate(name, size); e != nil {
		return &PathError{"truncate", name, e}
	}
	return nil
}

// Remove removes the named file or directory.
// If there is an error, it will be of type *PathError.
func Remove(name string) error {
	// System call interface forces us to know
	// whether name is a file or directory.
	// Try both: it is cheaper on average than
	// doing a Stat plus the right one.
	e := syscall.Unlink(name)
	if e == nil {
		return nil
	}
	e1 := syscall.Rmdir(name)
	if e1 == nil {
		return nil
	}

	// Both failed: figure out which error to return.
	// OS X and Linux differ on whether unlink(dir)
	// returns EISDIR, so can't use that.  However,
	// both agree that rmdir(file) returns ENOTDIR,
	// so we can use that to decide which error is real.
	// Rmdir might also return ENOTDIR if given a bad
	// file path, like /etc/passwd/foo, but in that case,
	// both errors will be ENOTDIR, so it's okay to
	// use the error from unlink.
	if e1 != syscall.ENOTDIR {
		e = e1
	}
	return &PathError{"remove", name, e}
}

// basename removes trailing slashes and the leading directory name from path name
func basename(name string) string {
	i := len(name) - 1
	// Remove trailing slashes
	for ; i > 0 && name[i] == '/'; i-- {
		name = name[:i]
	}
	// Remove leading directory name
	for i--; i >= 0; i-- {
		if name[i] == '/' {
			name = name[i+1:]
			break
		}
	}

	return name
}

// TempDir returns the default directory to use for temporary files.
func TempDir() string {
	dir := Getenv("TMPDIR")
	if dir == "" {
		if runtime.GOOS == "android" {
			dir = "/data/local/tmp"
		} else {
			dir = "/tmp"
		}
	}
	return dir
}

// Link creates newname as a hard link to the oldname file.
// If there is an error, it will be of type *LinkError.
func Link(oldname, newname string) error {
	e := syscall.Link(oldname, newname)
	if e != nil {
		return &LinkError{"link", oldname, newname, e}
	}
	return nil
}

// Symlink creates newname as a symbolic link to oldname.
// If there is an error, it will be of type *LinkError.
func Symlink(oldname, newname string) error {
	e := syscall.Symlink(oldname, newname)
	if e != nil {
		return &LinkError{"symlink", oldname, newname, e}
	}
	return nil
}
