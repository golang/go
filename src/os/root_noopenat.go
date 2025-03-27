// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (js && wasm) || plan9

package os

import (
	"errors"
	"sync/atomic"
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

func rootRemove(r *Root, name string) error {
	if err := checkPathEscapesLstat(r, name); err != nil {
		return &PathError{Op: "removeat", Path: name, Err: err}
	}
	if err := Remove(joinPath(r.root.name, name)); err != nil {
		return &PathError{Op: "removeat", Path: name, Err: underlyingError(err)}
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
