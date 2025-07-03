// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (js && wasm) || plan9

package os

import (
	"errors"
	"internal/filepathlite"
	"internal/stringslite"
	"sync/atomic"
	"syscall"
	"time"
)

// root implementation for platforms with no openat.
// Currently plan9 and js.
type root struct {
	name   string
	closed atomic.Bool
}

// openRootNolog is OpenRoot.
func openRootNolog(name string) (*Root, error) {
	r, err := newRoot(name)
	if err != nil {
		return nil, &PathError{Op: "open", Path: name, Err: err}
	}
	return r, nil
}

// openRootInRoot is Root.OpenRoot.
func openRootInRoot(r *Root, name string) (*Root, error) {
	if err := checkPathEscapes(r, name); err != nil {
		return nil, &PathError{Op: "openat", Path: name, Err: err}
	}
	r, err := newRoot(joinPath(r.root.name, name))
	if err != nil {
		return nil, &PathError{Op: "openat", Path: name, Err: err}
	}
	return r, nil
}

// newRoot returns a new Root.
// If fd is not a directory, it closes it and returns an error.
func newRoot(name string) (*Root, error) {
	fi, err := Stat(name)
	if err != nil {
		return nil, err.(*PathError).Err
	}
	if !fi.IsDir() {
		return nil, errors.New("not a directory")
	}
	return &Root{&root{name: name}}, nil
}

func (r *root) Close() error {
	// For consistency with platforms where Root.Close closes a handle,
	// mark the Root as closed and return errors from future calls.
	r.closed.Store(true)
	return nil
}

func (r *root) Name() string {
	return r.name
}

// rootOpenFileNolog is Root.OpenFile.
func rootOpenFileNolog(r *Root, name string, flag int, perm FileMode) (*File, error) {
	if err := checkPathEscapes(r, name); err != nil {
		return nil, &PathError{Op: "openat", Path: name, Err: err}
	}
	f, err := openFileNolog(joinPath(r.root.name, name), flag, perm)
	if err != nil {
		return nil, &PathError{Op: "openat", Path: name, Err: underlyingError(err)}
	}
	return f, nil
}

func rootStat(r *Root, name string, lstat bool) (FileInfo, error) {
	var fi FileInfo
	var err error
	if lstat {
		err = checkPathEscapesLstat(r, name)
		if err == nil {
			fi, err = Lstat(joinPath(r.root.name, name))
		}
	} else {
		err = checkPathEscapes(r, name)
		if err == nil {
			fi, err = Stat(joinPath(r.root.name, name))
		}
	}
	if err != nil {
		return nil, &PathError{Op: "statat", Path: name, Err: underlyingError(err)}
	}
	return fi, nil
}

func rootChmod(r *Root, name string, mode FileMode) error {
	if err := checkPathEscapes(r, name); err != nil {
		return &PathError{Op: "chmodat", Path: name, Err: err}
	}
	if err := Chmod(joinPath(r.root.name, name), mode); err != nil {
		return &PathError{Op: "chmodat", Path: name, Err: underlyingError(err)}
	}
	return nil
}

func rootChown(r *Root, name string, uid, gid int) error {
	if err := checkPathEscapes(r, name); err != nil {
		return &PathError{Op: "chownat", Path: name, Err: err}
	}
	if err := Chown(joinPath(r.root.name, name), uid, gid); err != nil {
		return &PathError{Op: "chownat", Path: name, Err: underlyingError(err)}
	}
	return nil
}

func rootLchown(r *Root, name string, uid, gid int) error {
	if err := checkPathEscapesLstat(r, name); err != nil {
		return &PathError{Op: "lchownat", Path: name, Err: err}
	}
	if err := Lchown(joinPath(r.root.name, name), uid, gid); err != nil {
		return &PathError{Op: "lchownat", Path: name, Err: underlyingError(err)}
	}
	return nil
}

func rootChtimes(r *Root, name string, atime time.Time, mtime time.Time) error {
	if err := checkPathEscapes(r, name); err != nil {
		return &PathError{Op: "chtimesat", Path: name, Err: err}
	}
	if err := Chtimes(joinPath(r.root.name, name), atime, mtime); err != nil {
		return &PathError{Op: "chtimesat", Path: name, Err: underlyingError(err)}
	}
	return nil
}

func rootMkdir(r *Root, name string, perm FileMode) error {
	if err := checkPathEscapes(r, name); err != nil {
		return &PathError{Op: "mkdirat", Path: name, Err: err}
	}
	if err := Mkdir(joinPath(r.root.name, name), perm); err != nil {
		return &PathError{Op: "mkdirat", Path: name, Err: underlyingError(err)}
	}
	return nil
}

func rootMkdirAll(r *Root, name string, perm FileMode) error {
	// We only check for errPathEscapes here.
	// For errors such as ENOTDIR (a non-directory file appeared somewhere along the path),
	// we let MkdirAll generate the error.
	// MkdirAll will return a PathError referencing the exact location of the error,
	// and we want to preserve that property.
	if err := checkPathEscapes(r, name); err == errPathEscapes {
		return &PathError{Op: "mkdirat", Path: name, Err: err}
	}
	prefix := r.root.name + string(PathSeparator)
	if err := MkdirAll(prefix+name, perm); err != nil {
		if pe, ok := err.(*PathError); ok {
			pe.Op = "mkdirat"
			pe.Path = stringslite.TrimPrefix(pe.Path, prefix)
			return pe
		}
		return &PathError{Op: "mkdirat", Path: name, Err: underlyingError(err)}
	}
	return nil
}

func rootRemove(r *Root, name string) error {
	if err := checkPathEscapesLstat(r, name); err != nil {
		return &PathError{Op: "removeat", Path: name, Err: err}
	}
	if endsWithDot(name) {
		// We don't want to permit removing the root itself, so check for that.
		if filepathlite.Clean(name) == "." {
			return &PathError{Op: "removeat", Path: name, Err: errPathEscapes}
		}
	}
	if err := Remove(joinPath(r.root.name, name)); err != nil {
		return &PathError{Op: "removeat", Path: name, Err: underlyingError(err)}
	}
	return nil
}

func rootRemoveAll(r *Root, name string) error {
	if endsWithDot(name) {
		// Consistency with os.RemoveAll: Return EINVAL when trying to remove .
		return &PathError{Op: "RemoveAll", Path: name, Err: syscall.EINVAL}
	}
	if err := checkPathEscapesLstat(r, name); err != nil {
		if err == syscall.ENOTDIR {
			// Some intermediate path component is not a directory.
			// RemoveAll treats this as success (since the target doesn't exist).
			return nil
		}
		return &PathError{Op: "RemoveAll", Path: name, Err: err}
	}
	if err := RemoveAll(joinPath(r.root.name, name)); err != nil {
		return &PathError{Op: "RemoveAll", Path: name, Err: underlyingError(err)}
	}
	return nil
}

func rootReadlink(r *Root, name string) (string, error) {
	if err := checkPathEscapesLstat(r, name); err != nil {
		return "", &PathError{Op: "readlinkat", Path: name, Err: err}
	}
	name, err := Readlink(joinPath(r.root.name, name))
	if err != nil {
		return "", &PathError{Op: "readlinkat", Path: name, Err: underlyingError(err)}
	}
	return name, nil
}

func rootRename(r *Root, oldname, newname string) error {
	if err := checkPathEscapesLstat(r, oldname); err != nil {
		return &PathError{Op: "renameat", Path: oldname, Err: err}
	}
	if err := checkPathEscapesLstat(r, newname); err != nil {
		return &PathError{Op: "renameat", Path: newname, Err: err}
	}
	err := Rename(joinPath(r.root.name, oldname), joinPath(r.root.name, newname))
	if err != nil {
		return &LinkError{"renameat", oldname, newname, underlyingError(err)}
	}
	return nil
}

func rootLink(r *Root, oldname, newname string) error {
	if err := checkPathEscapesLstat(r, oldname); err != nil {
		return &PathError{Op: "linkat", Path: oldname, Err: err}
	}
	fullOldName := joinPath(r.root.name, oldname)
	if fs, err := Lstat(fullOldName); err == nil && fs.Mode()&ModeSymlink != 0 {
		return &PathError{Op: "linkat", Path: oldname, Err: errors.New("cannot create a hard link to a symlink")}
	}
	if err := checkPathEscapesLstat(r, newname); err != nil {
		return &PathError{Op: "linkat", Path: newname, Err: err}
	}
	err := Link(fullOldName, joinPath(r.root.name, newname))
	if err != nil {
		return &LinkError{"linkat", oldname, newname, underlyingError(err)}
	}
	return nil
}

func rootSymlink(r *Root, oldname, newname string) error {
	if err := checkPathEscapesLstat(r, newname); err != nil {
		return &PathError{Op: "symlinkat", Path: newname, Err: err}
	}
	err := Symlink(oldname, joinPath(r.root.name, newname))
	if err != nil {
		return &LinkError{"symlinkat", oldname, newname, underlyingError(err)}
	}
	return nil
}
