// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || windows || wasip1

package os

import (
	"runtime"
	"slices"
	"sync"
	"syscall"
	"time"
)

// root implementation for platforms with a function to open a file
// relative to a directory.
type root struct {
	name string

	// refs is incremented while an operation is using fd.
	// closed is set when Close is called.
	// fd is closed when closed is true and refs is 0.
	mu      sync.Mutex
	fd      sysfdType
	refs    int  // number of active operations
	closed  bool // set when closed
	cleanup runtime.Cleanup
}

func (r *root) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.closed && r.refs == 0 {
		syscall.Close(r.fd)
	}
	r.closed = true
	r.cleanup.Stop()
	return nil
}

func (r *root) incref() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return ErrClosed
	}
	r.refs++
	return nil
}

func (r *root) decref() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.refs <= 0 {
		panic("bad Root refcount")
	}
	r.refs--
	if r.closed && r.refs == 0 {
		syscall.Close(r.fd)
	}
}

func (r *root) Name() string {
	return r.name
}

func rootChmod(r *Root, name string, mode FileMode) error {
	_, err := doInRoot(r, name, nil, func(parent sysfdType, name string) (struct{}, error) {
		return struct{}{}, chmodat(parent, name, mode)
	})
	if err != nil {
		return &PathError{Op: "chmodat", Path: name, Err: err}
	}
	return nil
}

func rootChown(r *Root, name string, uid, gid int) error {
	_, err := doInRoot(r, name, nil, func(parent sysfdType, name string) (struct{}, error) {
		return struct{}{}, chownat(parent, name, uid, gid)
	})
	if err != nil {
		return &PathError{Op: "chownat", Path: name, Err: err}
	}
	return nil
}

func rootLchown(r *Root, name string, uid, gid int) error {
	_, err := doInRoot(r, name, nil, func(parent sysfdType, name string) (struct{}, error) {
		return struct{}{}, lchownat(parent, name, uid, gid)
	})
	if err != nil {
		return &PathError{Op: "lchownat", Path: name, Err: err}
	}
	return err
}

func rootChtimes(r *Root, name string, atime time.Time, mtime time.Time) error {
	_, err := doInRoot(r, name, nil, func(parent sysfdType, name string) (struct{}, error) {
		return struct{}{}, chtimesat(parent, name, atime, mtime)
	})
	if err != nil {
		return &PathError{Op: "chtimesat", Path: name, Err: err}
	}
	return err
}

func rootMkdir(r *Root, name string, perm FileMode) error {
	_, err := doInRoot(r, name, nil, func(parent sysfdType, name string) (struct{}, error) {
		return struct{}{}, mkdirat(parent, name, perm)
	})
	if err != nil {
		return &PathError{Op: "mkdirat", Path: name, Err: err}
	}
	return nil
}

func rootMkdirAll(r *Root, fullname string, perm FileMode) error {
	// doInRoot opens each path element in turn.
	//
	// openDirFunc opens all but the last path component.
	// The usual default openDirFunc just opens directories with O_DIRECTORY.
	// We replace it here with one that creates missing directories along the way.
	openDirFunc := func(parent sysfdType, name string) (sysfdType, error) {
		for try := range 2 {
			fd, err := rootOpenDir(parent, name)
			switch err.(type) {
			case nil, errSymlink:
				return fd, err
			}
			if try > 0 || !IsNotExist(err) {
				return 0, &PathError{Op: "openat", Err: err}
			}
			if err := mkdirat(parent, name, perm); err != nil {
				return 0, &PathError{Op: "mkdirat", Err: err}
			}
		}
		panic("unreachable")
	}
	// openLastComponentFunc opens the last path component.
	openLastComponentFunc := func(parent sysfdType, name string) (struct{}, error) {
		err := mkdirat(parent, name, perm)
		if err == syscall.EEXIST {
			mode, e := modeAt(parent, name)
			if e == nil {
				if mode.IsDir() {
					// The target of MkdirAll is an existing directory.
					err = nil
				} else if mode&ModeSymlink != 0 {
					// The target of MkdirAll is a symlink.
					// For consistency with os.MkdirAll,
					// succeed if the link resolves to a directory.
					// We don't return errSymlink here, because we don't
					// want to create the link target if it doesn't exist.
					fi, e := r.Stat(fullname)
					if e == nil && fi.Mode().IsDir() {
						err = nil
					}
				}
			}
		}
		switch err.(type) {
		case nil, errSymlink:
			return struct{}{}, err
		}
		return struct{}{}, &PathError{Op: "mkdirat", Err: err}
	}
	_, err := doInRoot(r, fullname, openDirFunc, openLastComponentFunc)
	if err != nil {
		if _, ok := err.(*PathError); !ok {
			err = &PathError{Op: "mkdirat", Path: fullname, Err: err}
		}
	}
	return err
}

func rootReadlink(r *Root, name string) (string, error) {
	target, err := doInRoot(r, name, nil, func(parent sysfdType, name string) (string, error) {
		return readlinkat(parent, name)
	})
	if err != nil {
		return "", &PathError{Op: "readlinkat", Path: name, Err: err}
	}
	return target, nil
}

func rootRemove(r *Root, name string) error {
	_, err := doInRoot(r, name, nil, func(parent sysfdType, name string) (struct{}, error) {
		return struct{}{}, removeat(parent, name)
	})
	if err != nil {
		return &PathError{Op: "removeat", Path: name, Err: err}
	}
	return nil
}

func rootRemoveAll(r *Root, name string) error {
	// Consistency with os.RemoveAll: Strip trailing /s from the name,
	// so RemoveAll("not_a_directory/") succeeds.
	for len(name) > 0 && IsPathSeparator(name[len(name)-1]) {
		name = name[:len(name)-1]
	}
	if endsWithDot(name) {
		// Consistency with os.RemoveAll: Return EINVAL when trying to remove .
		return &PathError{Op: "RemoveAll", Path: name, Err: syscall.EINVAL}
	}
	_, err := doInRoot(r, name, nil, func(parent sysfdType, name string) (struct{}, error) {
		return struct{}{}, removeAllFrom(parent, name)
	})
	if IsNotExist(err) {
		return nil
	}
	if err != nil {
		return &PathError{Op: "RemoveAll", Path: name, Err: underlyingError(err)}
	}
	return err
}

func rootRename(r *Root, oldname, newname string) error {
	_, err := doInRoot(r, oldname, nil, func(oldparent sysfdType, oldname string) (struct{}, error) {
		_, err := doInRoot(r, newname, nil, func(newparent sysfdType, newname string) (struct{}, error) {
			return struct{}{}, renameat(oldparent, oldname, newparent, newname)
		})
		return struct{}{}, err
	})
	if err != nil {
		return &LinkError{"renameat", oldname, newname, err}
	}
	return err
}

func rootLink(r *Root, oldname, newname string) error {
	_, err := doInRoot(r, oldname, nil, func(oldparent sysfdType, oldname string) (struct{}, error) {
		_, err := doInRoot(r, newname, nil, func(newparent sysfdType, newname string) (struct{}, error) {
			return struct{}{}, linkat(oldparent, oldname, newparent, newname)
		})
		return struct{}{}, err
	})
	if err != nil {
		return &LinkError{"linkat", oldname, newname, err}
	}
	return err
}

// doInRoot performs an operation on a path in a Root.
//
// It calls f with the FD or handle for the directory containing the last
// path element, and the name of the last path element.
//
// For example, given the path a/b/c it calls f with the FD for a/b and the name "c".
//
// If openDirFunc is non-nil, it is called to open intermediate path elements.
// For example, given the path a/b/c openDirFunc will be called to open a and a/b in turn.
//
// f or openDirFunc may return errSymlink to indicate that the path element is a symlink
// which should be followed. Note that this can result in f being called multiple times
// with different names. For example, give the path "link" which is a symlink to "target",
// f is called with the path "link", returns errSymlink("target"), and is called again with
// the path "target".
//
// If f or openDirFunc return a *PathError, doInRoot will set PathError.Path to the
// full path which caused the error.
func doInRoot[T any](r *Root, name string, openDirFunc func(parent sysfdType, name string) (sysfdType, error), f func(parent sysfdType, name string) (T, error)) (ret T, err error) {
	if err := r.root.incref(); err != nil {
		return ret, err
	}
	defer r.root.decref()

	parts, suffixSep, err := splitPathInRoot(name, nil, nil)
	if err != nil {
		return ret, err
	}
	if openDirFunc == nil {
		openDirFunc = rootOpenDir
	}

	rootfd := r.root.fd
	dirfd := rootfd
	defer func() {
		if dirfd != rootfd {
			syscall.Close(dirfd)
		}
	}()

	// When resolving .. path components, we restart path resolution from the root.
	// (We can't openat(dir, "..") to move up to the parent directory,
	// because dir may have moved since we opened it.)
	// To limit how many opens a malicious path can cause us to perform, we set
	// a limit on the total number of path steps and the total number of restarts
	// caused by .. components. If *both* limits are exceeded, we halt the operation.
	const maxSteps = 255
	const maxRestarts = 8

	i := 0
	steps := 0
	restarts := 0
	symlinks := 0
Loop:
	for {
		steps++
		if steps > maxSteps && restarts > maxRestarts {
			return ret, syscall.ENAMETOOLONG
		}

		if parts[i] == ".." {
			// Resolve one or more parent ("..") path components.
			//
			// Rewrite the original path,
			// removing the elements eliminated by ".." components,
			// and start over from the beginning.
			restarts++
			end := i + 1
			for end < len(parts) && parts[end] == ".." {
				end++
			}
			count := end - i
			if count > i {
				return ret, errPathEscapes
			}
			parts = slices.Delete(parts, i-count, end)
			if len(parts) == 0 {
				parts = []string{"."}
			}
			i = 0
			if dirfd != rootfd {
				syscall.Close(dirfd)
			}
			dirfd = rootfd
			continue
		}

		if i == len(parts)-1 {
			// This is the last path element.
			// Call f to decide what to do with it.
			// If f returns errSymlink, this element is a symlink
			// which should be followed.
			// suffixSep contains any trailing separator characters
			// which we rejoin to the final part at this time.
			ret, err = f(dirfd, parts[i]+suffixSep)
			if err == nil {
				return
			}
		} else {
			var fd sysfdType
			fd, err = openDirFunc(dirfd, parts[i])
			if err == nil {
				if dirfd != rootfd {
					syscall.Close(dirfd)
				}
				dirfd = fd
			}
		}

		switch e := err.(type) {
		case nil:
		case errSymlink:
			symlinks++
			if symlinks > rootMaxSymlinks {
				return ret, syscall.ELOOP
			}
			newparts, newSuffixSep, err := splitPathInRoot(string(e), parts[:i], parts[i+1:])
			if err != nil {
				return ret, err
			}
			if i == len(parts)-1 {
				// suffixSep contains any trailing path separator characters
				// in the link target.
				// If we are replacing the remainder of the path, retain these.
				// If we're replacing some intermediate component of the path,
				// ignore them, since intermediate components must always be
				// directories.
				suffixSep = newSuffixSep
			}
			if len(newparts) < i || !slices.Equal(parts[:i], newparts[:i]) {
				// Some component in the path which we have already traversed
				// has changed. We need to restart parsing from the root.
				i = 0
				if dirfd != rootfd {
					syscall.Close(dirfd)
				}
				dirfd = rootfd
			}
			parts = newparts
			continue Loop
		case *PathError:
			// This is strings.Join(parts[:i+1], PathSeparator).
			e.Path = parts[0]
			for _, part := range parts[1 : i+1] {
				e.Path += string(PathSeparator) + part
			}
			return ret, e
		default:
			return ret, err
		}

		i++
	}
}

// errSymlink reports that a file being operated on is actually a symlink,
// and the target of that symlink.
type errSymlink string

func (errSymlink) Error() string { panic("errSymlink is not user-visible") }
