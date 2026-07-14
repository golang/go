// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || windows || wasip1

package os

import (
	"io/fs"
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
	mu     sync.Mutex
	fd     sysfdType
	refs   int  // number of active operations
	closed bool // set when closed
}

func (r *root) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.closed && r.refs == 0 {
		syscall.Close(r.fd)
	}
	r.closed = true
	runtime.SetFinalizer(r, nil) // no need for a finalizer any more
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
	_, err := doInRoot(r, name, 0, nil, func(parent sysfdType, name string, endsInSlash bool) (struct{}, error) {
		return struct{}{}, chmodat(parent, name, mode)
	})
	if err != nil {
		return &PathError{Op: "chmodat", Path: name, Err: err}
	}
	return nil
}

func rootChown(r *Root, name string, uid, gid int) error {
	_, err := doInRoot(r, name, 0, nil, func(parent sysfdType, name string, endsInSlash bool) (struct{}, error) {
		return struct{}{}, chownat(parent, name, uid, gid)
	})
	if err != nil {
		return &PathError{Op: "chownat", Path: name, Err: err}
	}
	return nil
}

func rootLchown(r *Root, name string, uid, gid int) error {
	_, err := doInRoot(r, name, 0, nil, func(parent sysfdType, name string, endsInSlash bool) (struct{}, error) {
		return struct{}{}, lchownat(parent, name, uid, gid)
	})
	if err != nil {
		return &PathError{Op: "lchownat", Path: name, Err: err}
	}
	return nil
}

func rootChtimes(r *Root, name string, atime time.Time, mtime time.Time) error {
	_, err := doInRoot(r, name, 0, nil, func(parent sysfdType, name string, endsInSlash bool) (struct{}, error) {
		return struct{}{}, chtimesat(parent, name, atime, mtime)
	})
	if err != nil {
		return &PathError{Op: "chtimesat", Path: name, Err: err}
	}
	return nil
}

func rootMkdir(r *Root, name string, perm FileMode) error {
	flags := uint(doInRootCreatingDirectory)
	switch runtime.GOOS {
	case "linux", "windows", "openbsd":
		// These platforms do not follow "symlink" on "mkdir symlink/".
		// (POSIX.1-2024 4.16 says that the trailing slash should cause
		// resolution to follow the symlink, but we're trying to match
		// platform semantics, not implement POSIX.)
		flags |= doInRootNoHandleTerminalSlash
	}
	_, err := doInRoot(r, name, flags, nil, func(parent sysfdType, name string, endsInSlash bool) (struct{}, error) {
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
			// Try again on EEXIST, because the directory may have been created
			// by another process or thread between the rootOpenDir and mkdirat calls.
			if err := mkdirat(parent, name, perm); err != nil && err != syscall.EEXIST {
				return 0, &PathError{Op: "mkdirat", Err: err}
			}
		}
		panic("unreachable")
	}
	// openLastComponentFunc opens the last path component.
	openLastComponentFunc := func(parent sysfdType, name string, endsInSlash bool) (struct{}, error) {
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
					} else if e == nil {
						err = syscall.ENOTDIR
					} else if !IsNotExist(e) {
						// EPERM, ELOOP, etc.,
						// probably more useful than EEXIST.
						err = e
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
	flags := uint(doInRootCreatingDirectory)
	switch runtime.GOOS {
	case "linux", "windows", "openbsd":
		flags |= doInRootNoHandleTerminalSlash // see rootMkdir
	}
	_, err := doInRoot(r, fullname, flags, openDirFunc, openLastComponentFunc)
	if err != nil {
		if _, ok := err.(*PathError); !ok {
			err = &PathError{Op: "mkdirat", Path: fullname, Err: err}
		}
	}
	return err
}

func rootReadlink(r *Root, name string) (string, error) {
	target, err := doInRoot(r, name, 0, nil, func(parent sysfdType, name string, endsInSlash bool) (string, error) {
		return readlinkat(parent, name)
	})
	if err != nil {
		return "", &PathError{Op: "readlinkat", Path: name, Err: err}
	}
	return target, nil
}

func rootRemove(r *Root, name string) error {
	_, err := doInRoot(r, name, 0, nil, func(parent sysfdType, name string, endsInSlash bool) (struct{}, error) {
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
	_, err := doInRoot(r, name, 0, nil, func(parent sysfdType, name string, endsInSlash bool) (struct{}, error) {
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
	_, err := doInRoot(r, oldname, 0, nil, func(oldparent sysfdType, oldname string, oldEndsInSlash bool) (struct{}, error) {
		flags := uint(doInRootCreatingDirectory)
		if runtime.GOOS == "windows" {
			flags |= doInRootNoHandleTerminalSlash
		}
		_, err := doInRoot(r, newname, flags, nil, func(newparent sysfdType, newname string, newEndsInSlash bool) (struct{}, error) {
			if runtime.GOOS != "windows" && newEndsInSlash {
				oldMode, err := modeAt(oldparent, oldname)
				if err != nil {
					return struct{}{}, err
				}
				if oldMode.Type() != fs.ModeDir {
					return struct{}{}, syscall.ENOTDIR
				}
			}
			// Same checks as applied by rename (in file_unix.go):
			fi, err := lstatat(newparent, newname)
			if err == nil && fi.IsDir() {
				if ofi, err := lstatat(oldparent, oldname); err != nil {
					return struct{}{}, err
				} else if newname == oldname || !SameFile(fi, ofi) {
					return struct{}{}, syscall.EEXIST
				}
			}
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
	_, err := doInRoot(r, oldname, 0, nil, func(oldparent sysfdType, oldname string, oldEndsInSlash bool) (struct{}, error) {
		flags := uint(0)
		if runtime.GOOS == "windows" {
			// Windows doesn't pay attention to trailing slashes in the link target.
			flags |= doInRootNoHandleTerminalSlash
		}
		_, err := doInRoot(r, newname, flags, nil, func(newparent sysfdType, newname string, newEndsInSlash bool) (struct{}, error) {
			return struct{}{}, linkat(oldparent, oldname, newparent, newname)
		})
		return struct{}{}, err
	})
	if err != nil {
		return &LinkError{"linkat", oldname, newname, err}
	}
	return err
}

// Flags for doInRoot.
const (
	// doInRootNoHandleTerminalSlash prevents doInRoot from applying special handling
	// for paths which end in one or more slashes.
	doInRootNoHandleTerminalSlash = 1 << iota

	// doInRootCreatingDirectory indicates that the operation is creating a directory.
	// When a path ends in /, the last path component may name a file which does not exist.
	doInRootCreatingDirectory

	// doInRootAlwaysResolveTerminalSlash causes doInRoot to resolve symlinks in the last
	// path component when a path ends in /, even on Windows. For example, this causes
	// doInRoot to resolve "symlink/" as the link target of "symlink".
	//
	// POSIX path operations resolve symlinks in this case.
	// Most Windows operations do not.
	// This flag enforces the POSIX behavior.
	doInRootAlwaysResolveTerminalSlash
)

// doInRoot performs an operation on a path in a Root.
//
// It calls f with the FD or handle for the directory containing the last
// path element, the name of the last path element (not including slashes),
// and a boolean indicating whether the original path ended in one or more slashes.
//
// For example, given the path a/b/c it calls f with the FD for a/b and the name "c".
//
// It applies special handling for paths ending in a slash: When a path ends in a slash
// (for example "a/b/"), doInRoot will check the final component ("b") before calling f.
// If the final component is a symlink, doInRoot will resolve it.
// If the final component is neither a symlink nor a directory, doInRoot will return ENOTDIR.
// This behavior may be disabled by passing the doInRootNoHandleTerminalSlash flag.
//
// If openDirFunc is non-nil, it is called to open intermediate path elements.
// For example, given the path a/b/c openDirFunc will be called to open a and a/b in turn.
//
// f or openDirFunc may return errSymlink to indicate that the path element is a symlink
// which should be followed. Note that this can result in f being called multiple times
// with different names. For example, given the path "link" which is a symlink to "target",
// f is called with the path "link", returns errSymlink("target"), and is called again with
// the path "target".
//
// If f or openDirFunc return a *PathError, doInRoot will set PathError.Path to the
// full path which caused the error.
func doInRoot[T any](r *Root, name string, flags uint, openDirFunc func(parent sysfdType, name string) (sysfdType, error), f func(parent sysfdType, name string, endsInSlash bool) (T, error)) (ret T, err error) {
	if err := r.root.incref(); err != nil {
		return ret, err
	}
	defer r.root.decref()

	parts, endsInSlash, err := splitPathInRoot(name, nil, nil)
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
			err = nil
			if endsInSlash && flags&doInRootNoHandleTerminalSlash == 0 {
				var fi FileInfo
				fi, err = lstatat(dirfd, parts[i])
				switch {
				case IsNotExist(err) && flags&doInRootCreatingDirectory != 0:
					// The path ends in a slash, the last path component
					// does not exist, and we creating a directory.
					// This is fine.
					err = nil
				case err != nil:
					return
				case fi.Mode().Type() == fs.ModeDir:
				case fi.Mode().Type() == fs.ModeSymlink:
					if runtime.GOOS != "windows" || flags&doInRootAlwaysResolveTerminalSlash != 0 {
						err = checkSymlink(dirfd, parts[i], syscall.ENOTDIR)
					} else {
						if !isDirectoryLink(fi) {
							err = syscall.ENOTDIR
						}
					}
				default:
					err = syscall.ENOTDIR
					return
				}
			}

			// This is the last path element.
			// Call f to decide what to do with it.
			// If f returns errSymlink, this element is a symlink
			// which should be followed.
			// suffixSep contains any trailing separator characters.
			if err == nil {
				ret, err = f(dirfd, parts[i], endsInSlash)
				if err == nil {
					return
				}
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
			lastPart := i == len(parts)-1
			newparts, newEndsInSlash, err := splitPathInRoot(string(e), parts[:i], parts[i+1:])
			if err != nil {
				return ret, err
			}
			if lastPart && newEndsInSlash {
				// If a link target in the final path component ends in a slash,
				// then the path now ends in a slash.
				endsInSlash = true
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

func modeAt(parent sysfdType, name string) (FileMode, error) {
	fi, err := lstatat(parent, name)
	if err != nil {
		return 0, err
	}
	return fi.Mode(), nil
}

// errSymlink reports that a file being operated on is actually a symlink,
// and the target of that symlink.
type errSymlink string

func (errSymlink) Error() string { panic("errSymlink is not user-visible") }
