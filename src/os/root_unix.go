// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || wasip1

package os

import (
	"errors"
	"internal/syscall/unix"
	"runtime"
	"syscall"
	"time"
)

// sysfdType is the native type of a file handle
// (int on Unix, syscall.Handle on Windows),
// permitting helper functions to be written portably.
type sysfdType = int

// openRootNolog is OpenRoot.
func openRootNolog(name string) (*Root, error) {
	var fd int
	err := ignoringEINTR(func() error {
		var err error
		fd, _, err = open(name, syscall.O_CLOEXEC, 0)
		return err
	})
	if err != nil {
		return nil, &PathError{Op: "open", Path: name, Err: err}
	}
	return newRoot(fd, name)
}

// newRoot returns a new Root.
// If fd is not a directory, it closes it and returns an error.
func newRoot(fd int, name string) (*Root, error) {
	var fs fileStat
	err := ignoringEINTR(func() error {
		return syscall.Fstat(fd, &fs.sys)
	})
	fillFileStatFromSys(&fs, name)
	if err == nil && !fs.IsDir() {
		syscall.Close(fd)
		return nil, &PathError{Op: "open", Path: name, Err: errors.New("not a directory")}
	}

	// There's a race here with fork/exec, which we are
	// content to live with. See ../syscall/exec_unix.go.
	if !supportsCloseOnExec {
		syscall.CloseOnExec(fd)
	}

	r := &Root{&root{
		fd:   fd,
		name: name,
	}}
	r.root.cleanup = runtime.AddCleanup(r, func(f *root) { f.Close() }, r.root)
	return r, nil
}

// openRootInRoot is Root.OpenRoot.
func openRootInRoot(r *Root, name string) (*Root, error) {
	fd, err := doInRoot(r, name, nil, func(parent int, name string) (fd int, err error) {
		ignoringEINTR(func() error {
			fd, err = unix.Openat(parent, name, syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
			if isNoFollowErr(err) {
				err = checkSymlink(parent, name, err)
			}
			return err
		})
		return fd, err
	})
	if err != nil {
		return nil, &PathError{Op: "openat", Path: name, Err: err}
	}
	return newRoot(fd, name)
}

// rootOpenFileNolog is Root.OpenFile.
func rootOpenFileNolog(root *Root, name string, flag int, perm FileMode) (*File, error) {
	fd, err := doInRoot(root, name, nil, func(parent int, name string) (fd int, err error) {
		ignoringEINTR(func() error {
			fd, err = unix.Openat(parent, name, syscall.O_NOFOLLOW|syscall.O_CLOEXEC|flag, uint32(perm))
			if isNoFollowErr(err) || err == syscall.ENOTDIR {
				err = checkSymlink(parent, name, err)
			}
			return err
		})
		return fd, err
	})
	if err != nil {
		return nil, &PathError{Op: "openat", Path: name, Err: err}
	}
	f := newFile(fd, joinPath(root.Name(), name), kindOpenFile, unix.HasNonblockFlag(flag))
	return f, nil
}

func rootOpenDir(parent int, name string) (int, error) {
	var (
		fd  int
		err error
	)
	ignoringEINTR(func() error {
		fd, err = unix.Openat(parent, name, syscall.O_NOFOLLOW|syscall.O_CLOEXEC|syscall.O_DIRECTORY, 0)
		if isNoFollowErr(err) || err == syscall.ENOTDIR {
			err = checkSymlink(parent, name, err)
		} else if err == syscall.ENOTSUP || err == syscall.EOPNOTSUPP {
			// ENOTSUP and EOPNOTSUPP are often, but not always, the same errno.
			// Translate both to ENOTDIR, since this indicates a non-terminal
			// path component was not a directory.
			err = syscall.ENOTDIR
		}
		return err
	})
	return fd, err
}

func rootStat(r *Root, name string, lstat bool) (FileInfo, error) {
	fi, err := doInRoot(r, name, nil, func(parent sysfdType, n string) (FileInfo, error) {
		var fs fileStat
		if err := unix.Fstatat(parent, n, &fs.sys, unix.AT_SYMLINK_NOFOLLOW); err != nil {
			return nil, err
		}
		fillFileStatFromSys(&fs, name)
		if !lstat && fs.Mode()&ModeSymlink != 0 {
			return nil, checkSymlink(parent, n, syscall.ELOOP)
		}
		return &fs, nil
	})
	if err != nil {
		return nil, &PathError{Op: "statat", Path: name, Err: err}
	}
	return fi, nil
}

func rootSymlink(r *Root, oldname, newname string) error {
	_, err := doInRoot(r, newname, nil, func(parent sysfdType, name string) (struct{}, error) {
		return struct{}{}, symlinkat(oldname, parent, name)
	})
	if err != nil {
		return &LinkError{Op: "symlinkat", Old: oldname, New: newname, Err: err}
	}
	return nil
}

// On systems which use fchmodat, fchownat, etc., we have a race condition:
// When "name" is a symlink, Root.Chmod("name") should act on the target of that link.
// However, fchmodat doesn't allow us to chmod a file only if it is not a symlink;
// the AT_SYMLINK_NOFOLLOW parameter causes the operation to act on the symlink itself.
//
// We do the best we can by first checking to see if the target of the operation is a symlink,
// and only attempting the fchmodat if it is not. If the target is replaced between the check
// and the fchmodat, we will chmod the symlink rather than following it.
//
// This race condition is unfortunate, but does not permit escaping a root:
// We may act on the wrong file, but that file will be contained within the root.
func afterResolvingSymlink(parent int, name string, f func() error) error {
	if err := checkSymlink(parent, name, nil); err != nil {
		return err
	}
	return f()
}

func chmodat(parent int, name string, mode FileMode) error {
	return afterResolvingSymlink(parent, name, func() error {
		return ignoringEINTR(func() error {
			return unix.Fchmodat(parent, name, syscallMode(mode), unix.AT_SYMLINK_NOFOLLOW)
		})
	})
}

func chownat(parent int, name string, uid, gid int) error {
	return afterResolvingSymlink(parent, name, func() error {
		return ignoringEINTR(func() error {
			return unix.Fchownat(parent, name, uid, gid, unix.AT_SYMLINK_NOFOLLOW)
		})
	})
}

func lchownat(parent int, name string, uid, gid int) error {
	return ignoringEINTR(func() error {
		return unix.Fchownat(parent, name, uid, gid, unix.AT_SYMLINK_NOFOLLOW)
	})
}

func chtimesat(parent int, name string, atime time.Time, mtime time.Time) error {
	return afterResolvingSymlink(parent, name, func() error {
		return ignoringEINTR(func() error {
			utimes := chtimesUtimes(atime, mtime)
			return unix.Utimensat(parent, name, &utimes, unix.AT_SYMLINK_NOFOLLOW)
		})
	})
}

func mkdirat(fd int, name string, perm FileMode) error {
	return ignoringEINTR(func() error {
		return unix.Mkdirat(fd, name, syscallMode(perm))
	})
}

func removeat(fd int, name string) error {
	// The system call interface forces us to know whether
	// we are removing a file or directory. Try both.
	e := ignoringEINTR(func() error {
		return unix.Unlinkat(fd, name, 0)
	})
	if e == nil {
		return nil
	}
	e1 := ignoringEINTR(func() error {
		return unix.Unlinkat(fd, name, unix.AT_REMOVEDIR)
	})
	if e1 == nil {
		return nil
	}
	// Both failed. See comment in Remove for how we decide which error to return.
	if e1 != syscall.ENOTDIR {
		return e1
	}
	return e
}

func removefileat(fd int, name string) error {
	return ignoringEINTR(func() error {
		return unix.Unlinkat(fd, name, 0)
	})
}

func removedirat(fd int, name string) error {
	return ignoringEINTR(func() error {
		return unix.Unlinkat(fd, name, unix.AT_REMOVEDIR)
	})
}

func renameat(oldfd int, oldname string, newfd int, newname string) error {
	return unix.Renameat(oldfd, oldname, newfd, newname)
}

func linkat(oldfd int, oldname string, newfd int, newname string) error {
	return unix.Linkat(oldfd, oldname, newfd, newname, 0)
}

func symlinkat(oldname string, newfd int, newname string) error {
	return unix.Symlinkat(oldname, newfd, newname)
}

func modeAt(parent int, name string) (FileMode, error) {
	var fs fileStat
	if err := unix.Fstatat(parent, name, &fs.sys, unix.AT_SYMLINK_NOFOLLOW); err != nil {
		return 0, err
	}
	fillFileStatFromSys(&fs, name)
	return fs.mode, nil
}

// checkSymlink resolves the symlink name in parent,
// and returns errSymlink with the link contents.
//
// If name is not a symlink, return origError.
func checkSymlink(parent int, name string, origError error) error {
	link, err := readlinkat(parent, name)
	if err != nil {
		return origError
	}
	return errSymlink(link)
}

func readlinkat(fd int, name string) (string, error) {
	for len := 128; ; len *= 2 {
		b := make([]byte, len)
		var (
			n int
			e error
		)
		ignoringEINTR(func() error {
			n, e = unix.Readlinkat(fd, name, b)
			return e
		})
		if e == syscall.ERANGE {
			continue
		}
		if e != nil {
			return "", e
		}
		n = max(n, 0)
		if n < len {
			return string(b[0:n]), nil
		}
	}
}
