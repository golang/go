// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The os package provides a platform-independent interface to operating
// system functionality.  The design is Unix-like.
package os

import (
	"syscall";
)

// Auxiliary information if the File describes a directory
type dirInfo struct {
	buf	[]byte;	// buffer for directory I/O
	nbuf	int;	// length of buf; return value from Getdirentries
	bufp	int;	// location of next record in buf.
}

// File represents an open file descriptor.
type File struct {
	fd	int;
	name	string;
	dirinfo	*dirInfo;	// nil unless directory being read
	nepipe	int;		// number of consecutive EPIPE in Write
}

// Fd returns the integer Unix file descriptor referencing the open file.
func (file *File) Fd() int	{ return file.fd }

// Name returns the name of the file as presented to Open.
func (file *File) Name() string	{ return file.name }

// NewFile returns a new File with the given file descriptor and name.
func NewFile(fd int, name string) *File {
	if fd < 0 {
		return nil
	}
	return &File{fd, name, nil, 0};
}

// Stdin, Stdout, and Stderr are open Files pointing to the standard input,
// standard output, and standard error file descriptors.
var (
	Stdin	= NewFile(0, "/dev/stdin");
	Stdout	= NewFile(1, "/dev/stdout");
	Stderr	= NewFile(2, "/dev/stderr");
)

// Flags to Open wrapping those of the underlying system. Not all flags
// may be implemented on a given system.
const (
	O_RDONLY	= syscall.O_RDONLY;	// open the file read-only.
	O_WRONLY	= syscall.O_WRONLY;	// open the file write-only.
	O_RDWR		= syscall.O_RDWR;	// open the file read-write.
	O_APPEND	= syscall.O_APPEND;	// open the file append-only.
	O_ASYNC		= syscall.O_ASYNC;	// generate a signal when I/O is available.
	O_CREAT		= syscall.O_CREAT;	// create a new file if none exists.
	O_EXCL		= syscall.O_EXCL;	// used with O_CREAT, file must not exist
	O_NOCTTY	= syscall.O_NOCTTY;	// do not make file the controlling tty.
	O_NONBLOCK	= syscall.O_NONBLOCK;	// open in non-blocking mode.
	O_NDELAY	= O_NONBLOCK;		// synonym for O_NONBLOCK
	O_SYNC		= syscall.O_SYNC;	// open for synchronous I/O.
	O_TRUNC		= syscall.O_TRUNC;	// if possible, truncate file when opened.
	O_CREATE	= O_CREAT;		// create a new file if none exists.
)

// Open opens the named file with specified flag (O_RDONLY etc.) and perm, (0666 etc.)
// if applicable.  If successful, methods on the returned File can be used for I/O.
// It returns the File and an Error, if any.
func Open(name string, flag int, perm int) (file *File, err Error) {
	r, e := syscall.Open(name, flag|syscall.O_CLOEXEC, perm);
	if e != 0 {
		return nil, &PathError{"open", name, Errno(e)}
	}

	// There's a race here with fork/exec, which we are
	// content to live with.  See ../syscall/exec.go
	if syscall.O_CLOEXEC == 0 {	// O_CLOEXEC not supported
		syscall.CloseOnExec(r)
	}

	return NewFile(r, name), nil;
}

// Close closes the File, rendering it unusable for I/O.
// It returns an Error, if any.
func (file *File) Close() Error {
	if file == nil {
		return EINVAL
	}
	var err Error;
	if e := syscall.Close(file.fd); e != 0 {
		err = &PathError{"close", file.name, Errno(e)}
	}
	file.fd = -1;	// so it can't be closed again
	return err;
}

type eofError int

func (eofError) String() string	{ return "EOF" }

// EOF is the Error returned by Read when no more input is available.
// Functions should return EOF only to signal a graceful end of input.
// If the EOF occurs unexpectedly in a structured data stream,
// the appropriate error is either io.ErrUnexpectedEOF or some other error
// giving more detail.
var EOF Error = eofError(0)

// Read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an Error, if any.
// EOF is signaled by a zero count with err set to EOF.
func (file *File) Read(b []byte) (n int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	n, e := syscall.Read(file.fd, b);
	if n < 0 {
		n = 0
	}
	if n == 0 && e == 0 {
		return 0, EOF
	}
	if e != 0 {
		err = &PathError{"read", file.name, Errno(e)}
	}
	return n, err;
}

// ReadAt reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the Error, if any.
// EOF is signaled by a zero count with err set to EOF.
// ReadAt always returns a non-nil Error when n != len(b).
func (file *File) ReadAt(b []byte, off int64) (n int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	for len(b) > 0 {
		m, e := syscall.Pread(file.fd, b, off);
		n += m;
		if e != 0 {
			err = &PathError{"read", file.name, Errno(e)};
			break;
		}
		b = b[m:len(b)];
		off += int64(m);
	}
	return;
}

// Write writes len(b) bytes to the File.
// It returns the number of bytes written and an Error, if any.
// Write returns a non-nil Error when n != len(b).
func (file *File) Write(b []byte) (n int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	n, e := syscall.Write(file.fd, b);
	if n < 0 {
		n = 0
	}
	if e == syscall.EPIPE {
		file.nepipe++;
		if file.nepipe >= 10 {
			Exit(syscall.EPIPE)
		}
	} else {
		file.nepipe = 0
	}
	if e != 0 {
		err = &PathError{"write", file.name, Errno(e)}
	}
	return n, err;
}

// WriteAt writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an Error, if any.
// WriteAt returns a non-nil Error when n != len(b).
func (file *File) WriteAt(b []byte, off int64) (n int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	for len(b) > 0 {
		m, e := syscall.Pwrite(file.fd, b, off);
		n += m;
		if e != 0 {
			err = &PathError{"write", file.name, Errno(e)};
			break;
		}
		b = b[m:len(b)];
		off += int64(m);
	}
	return;
}

// Seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an Error, if any.
func (file *File) Seek(offset int64, whence int) (ret int64, err Error) {
	r, e := syscall.Seek(file.fd, offset, whence);
	if e == 0 && file.dirinfo != nil && r != 0 {
		e = syscall.EISDIR
	}
	if e != 0 {
		return 0, &PathError{"seek", file.name, Errno(e)}
	}
	return r, nil;
}

// WriteString is like Write, but writes the contents of string s rather than
// an array of bytes.
func (file *File) WriteString(s string) (ret int, err Error) {
	if file == nil {
		return 0, EINVAL
	}
	b := syscall.StringByteSlice(s);
	b = b[0 : len(b)-1];
	return file.Write(b);
}

// Pipe returns a connected pair of Files; reads from r return bytes written to w.
// It returns the files and an Error, if any.
func Pipe() (r *File, w *File, err Error) {
	var p [2]int;

	// See ../syscall/exec.go for description of lock.
	syscall.ForkLock.RLock();
	e := syscall.Pipe(&p);
	if e != 0 {
		syscall.ForkLock.RUnlock();
		return nil, nil, NewSyscallError("pipe", e);
	}
	syscall.CloseOnExec(p[0]);
	syscall.CloseOnExec(p[1]);
	syscall.ForkLock.RUnlock();

	return NewFile(p[0], "|0"), NewFile(p[1], "|1"), nil;
}

// Mkdir creates a new directory with the specified name and permission bits.
// It returns an error, if any.
func Mkdir(name string, perm int) Error {
	e := syscall.Mkdir(name, perm);
	if e != 0 {
		return &PathError{"mkdir", name, Errno(e)}
	}
	return nil;
}

// Stat returns a Dir structure describing the named file and an error, if any.
// If name names a valid symbolic link, the returned Dir describes
// the file pointed at by the link and has dir.FollowedSymlink set to true.
// If name names an invalid symbolic link, the returned Dir describes
// the link itself and has dir.FollowedSymlink set to false.
func Stat(name string) (dir *Dir, err Error) {
	var lstat, stat syscall.Stat_t;
	e := syscall.Lstat(name, &lstat);
	if e != 0 {
		return nil, &PathError{"stat", name, Errno(e)}
	}
	statp := &lstat;
	if lstat.Mode&syscall.S_IFMT == syscall.S_IFLNK {
		e := syscall.Stat(name, &stat);
		if e == 0 {
			statp = &stat
		}
	}
	return dirFromStat(name, new(Dir), &lstat, statp), nil;
}

// Stat returns the Dir structure describing file.
// It returns the Dir and an error, if any.
func (file *File) Stat() (dir *Dir, err Error) {
	var stat syscall.Stat_t;
	e := syscall.Fstat(file.fd, &stat);
	if e != 0 {
		return nil, &PathError{"stat", file.name, Errno(e)}
	}
	return dirFromStat(file.name, new(Dir), &stat, &stat), nil;
}

// Lstat returns the Dir structure describing the named file and an error, if any.
// If the file is a symbolic link, the returned Dir describes the
// symbolic link.  Lstat makes no attempt to follow the link.
func Lstat(name string) (dir *Dir, err Error) {
	var stat syscall.Stat_t;
	e := syscall.Lstat(name, &stat);
	if e != 0 {
		return nil, &PathError{"lstat", name, Errno(e)}
	}
	return dirFromStat(name, new(Dir), &stat, &stat), nil;
}

// Readdir reads the contents of the directory associated with file and
// returns an array of up to count Dir structures, as would be returned
// by Stat, in directory order.  Subsequent calls on the same file will yield further Dirs.
// A negative count means to read until EOF.
// Readdir returns the array and an Error, if any.
func (file *File) Readdir(count int) (dirs []Dir, err Error) {
	dirname := file.name;
	if dirname == "" {
		dirname = "."
	}
	dirname += "/";
	names, err1 := file.Readdirnames(count);
	if err1 != nil {
		return nil, err1
	}
	dirs = make([]Dir, len(names));
	for i, filename := range names {
		dirp, err := Lstat(dirname + filename);
		if dirp == nil || err != nil {
			dirs[i].Name = filename	// rest is already zeroed out
		} else {
			dirs[i] = *dirp
		}
	}
	return;
}

// Chdir changes the current working directory to the named directory.
func Chdir(dir string) Error {
	if e := syscall.Chdir(dir); e != 0 {
		return &PathError{"chdir", dir, Errno(e)}
	}
	return nil;
}

// Chdir changes the current working directory to the file,
// which must be a directory.
func (f *File) Chdir() Error {
	if e := syscall.Fchdir(f.fd); e != 0 {
		return &PathError{"chdir", f.name, Errno(e)}
	}
	return nil;
}

// Remove removes the named file or directory.
func Remove(name string) Error {
	// System call interface forces us to know
	// whether name is a file or directory.
	// Try both: it is cheaper on average than
	// doing a Stat plus the right one.
	e := syscall.Unlink(name);
	if e == 0 {
		return nil
	}
	e1 := syscall.Rmdir(name);
	if e1 == 0 {
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
	return &PathError{"remove", name, Errno(e)};
}

// LinkError records an error during a link or symlink
// system call and the paths that caused it.
type LinkError struct {
	Op	string;
	Old	string;
	New	string;
	Error	Error;
}

func (e *LinkError) String() string {
	return e.Op + " " + e.Old + " " + e.New + ": " + e.Error.String()
}

// Link creates a hard link.
func Link(oldname, newname string) Error {
	e := syscall.Link(oldname, newname);
	if e != 0 {
		return &LinkError{"link", oldname, newname, Errno(e)}
	}
	return nil;
}

// Symlink creates a symbolic link.
func Symlink(oldname, newname string) Error {
	e := syscall.Symlink(oldname, newname);
	if e != 0 {
		return &LinkError{"symlink", oldname, newname, Errno(e)}
	}
	return nil;
}

// Readlink reads the contents of a symbolic link: the destination of
// the link.  It returns the contents and an Error, if any.
func Readlink(name string) (string, Error) {
	for len := 128; ; len *= 2 {
		b := make([]byte, len);
		n, e := syscall.Readlink(name, b);
		if e != 0 {
			return "", &PathError{"readlink", name, Errno(e)}
		}
		if n < len {
			return string(b[0:n]), nil
		}
	}
	// Silence 6g.
	return "", nil;
}

// Chmod changes the mode of the named file to mode.
// If the file is a symbolic link, it changes the uid and gid of the link's target.
func Chmod(name string, mode int) Error {
	if e := syscall.Chmod(name, mode); e != 0 {
		return &PathError{"chmod", name, Errno(e)}
	}
	return nil;
}

// Chmod changes the mode of the file to mode.
func (f *File) Chmod(mode int) Error {
	if e := syscall.Fchmod(f.fd, mode); e != 0 {
		return &PathError{"chmod", f.name, Errno(e)}
	}
	return nil;
}

// Chown changes the numeric uid and gid of the named file.
// If the file is a symbolic link, it changes the uid and gid of the link's target.
func Chown(name string, uid, gid int) Error {
	if e := syscall.Chown(name, uid, gid); e != 0 {
		return &PathError{"chown", name, Errno(e)}
	}
	return nil;
}

// Lchown changes the numeric uid and gid of the named file.
// If the file is a symbolic link, it changes the uid and gid of the link itself.
func Lchown(name string, uid, gid int) Error {
	if e := syscall.Lchown(name, uid, gid); e != 0 {
		return &PathError{"lchown", name, Errno(e)}
	}
	return nil;
}

// Chown changes the numeric uid and gid of the named file.
func (f *File) Chown(uid, gid int) Error {
	if e := syscall.Fchown(f.fd, uid, gid); e != 0 {
		return &PathError{"chown", f.name, Errno(e)}
	}
	return nil;
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
func Truncate(name string, size int64) Error {
	if e := syscall.Truncate(name, size); e != 0 {
		return &PathError{"truncate", name, Errno(e)}
	}
	return nil;
}

// Truncate changes the size of the file.
// It does not change the I/O offset.
func (f *File) Truncate(size int64) Error {
	if e := syscall.Ftruncate(f.fd, size); e != 0 {
		return &PathError{"truncate", f.name, Errno(e)}
	}
	return nil;
}
