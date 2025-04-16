// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"errors"
	"internal/bytealg"
	"internal/stringslite"
	"internal/testlog"
	"io/fs"
	"runtime"
	"slices"
)

// OpenInRoot opens the file name in the directory dir.
// It is equivalent to OpenRoot(dir) followed by opening the file in the root.
//
// OpenInRoot returns an error if any component of the name
// references a location outside of dir.
//
// See [Root] for details and limitations.
func OpenInRoot(dir, name string) (*File, error) {
	r, err := OpenRoot(dir)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return r.Open(name)
}

// Root may be used to only access files within a single directory tree.
//
// Methods on Root can only access files and directories beneath a root directory.
// If any component of a file name passed to a method of Root references a location
// outside the root, the method returns an error.
// File names may reference the directory itself (.).
//
// Methods on Root will follow symbolic links, but symbolic links may not
// reference a location outside the root.
// Symbolic links must not be absolute.
//
// Methods on Root do not prohibit traversal of filesystem boundaries,
// Linux bind mounts, /proc special files, or access to Unix device files.
//
// Methods on Root are safe to be used from multiple goroutines simultaneously.
//
// On most platforms, creating a Root opens a file descriptor or handle referencing
// the directory. If the directory is moved, methods on Root reference the original
// directory in its new location.
//
// Root's behavior differs on some platforms:
//
//   - When GOOS=windows, file names may not reference Windows reserved device names
//     such as NUL and COM1.
//   - When GOOS=js, Root is vulnerable to TOCTOU (time-of-check-time-of-use)
//     attacks in symlink validation, and cannot ensure that operations will not
//     escape the root.
//   - When GOOS=plan9 or GOOS=js, Root does not track directories across renames.
//     On these platforms, a Root references a directory name, not a file descriptor.
type Root struct {
	root *root
}

const (
	// Maximum number of symbolic links we will follow when resolving a file in a root.
	// 8 is __POSIX_SYMLOOP_MAX (the minimum allowed value for SYMLOOP_MAX),
	// and a common limit.
	rootMaxSymlinks = 8
)

// OpenRoot opens the named directory.
// If there is an error, it will be of type *PathError.
func OpenRoot(name string) (*Root, error) {
	testlog.Open(name)
	return openRootNolog(name)
}

// Name returns the name of the directory presented to OpenRoot.
//
// It is safe to call Name after [Close].
func (r *Root) Name() string {
	return r.root.Name()
}

// Close closes the Root.
// After Close is called, methods on Root return errors.
func (r *Root) Close() error {
	return r.root.Close()
}

// Open opens the named file in the root for reading.
// See [Open] for more details.
func (r *Root) Open(name string) (*File, error) {
	return r.OpenFile(name, O_RDONLY, 0)
}

// Create creates or truncates the named file in the root.
// See [Create] for more details.
func (r *Root) Create(name string) (*File, error) {
	return r.OpenFile(name, O_RDWR|O_CREATE|O_TRUNC, 0666)
}

// OpenFile opens the named file in the root.
// See [OpenFile] for more details.
//
// If perm contains bits other than the nine least-significant bits (0o777),
// OpenFile returns an error.
func (r *Root) OpenFile(name string, flag int, perm FileMode) (*File, error) {
	if perm&0o777 != perm {
		return nil, &PathError{Op: "openat", Path: name, Err: errors.New("unsupported file mode")}
	}
	r.logOpen(name)
	rf, err := rootOpenFileNolog(r, name, flag, perm)
	if err != nil {
		return nil, err
	}
	rf.appendMode = flag&O_APPEND != 0
	return rf, nil
}

// OpenRoot opens the named directory in the root.
// If there is an error, it will be of type *PathError.
func (r *Root) OpenRoot(name string) (*Root, error) {
	r.logOpen(name)
	return openRootInRoot(r, name)
}

// Mkdir creates a new directory in the root
// with the specified name and permission bits (before umask).
// See [Mkdir] for more details.
//
// If perm contains bits other than the nine least-significant bits (0o777),
// OpenFile returns an error.
func (r *Root) Mkdir(name string, perm FileMode) error {
	if perm&0o777 != perm {
		return &PathError{Op: "mkdirat", Path: name, Err: errors.New("unsupported file mode")}
	}
	return rootMkdir(r, name, perm)
}

// Remove removes the named file or (empty) directory in the root.
// See [Remove] for more details.
func (r *Root) Remove(name string) error {
	return rootRemove(r, name)
}

// Stat returns a [FileInfo] describing the named file in the root.
// See [Stat] for more details.
func (r *Root) Stat(name string) (FileInfo, error) {
	r.logStat(name)
	return rootStat(r, name, false)
}

// Lstat returns a [FileInfo] describing the named file in the root.
// If the file is a symbolic link, the returned FileInfo
// describes the symbolic link.
// See [Lstat] for more details.
func (r *Root) Lstat(name string) (FileInfo, error) {
	r.logStat(name)
	return rootStat(r, name, true)
}

func (r *Root) logOpen(name string) {
	if log := testlog.Logger(); log != nil {
		// This won't be right if r's name has changed since it was opened,
		// but it's the best we can do.
		log.Open(joinPath(r.Name(), name))
	}
}

func (r *Root) logStat(name string) {
	if log := testlog.Logger(); log != nil {
		// This won't be right if r's name has changed since it was opened,
		// but it's the best we can do.
		log.Stat(joinPath(r.Name(), name))
	}
}

// splitPathInRoot splits a path into components
// and joins it with the given prefix and suffix.
//
// The path is relative to a Root, and must not be
// absolute, volume-relative, or "".
//
// "." components are removed, except in the last component.
//
// Path separators following the last component are returned in suffixSep.
func splitPathInRoot(s string, prefix, suffix []string) (_ []string, suffixSep string, err error) {
	if len(s) == 0 {
		return nil, "", errors.New("empty path")
	}
	if IsPathSeparator(s[0]) {
		return nil, "", errPathEscapes
	}

	if runtime.GOOS == "windows" {
		// Windows cleans paths before opening them.
		s, err = rootCleanPath(s, prefix, suffix)
		if err != nil {
			return nil, "", err
		}
		prefix = nil
		suffix = nil
	}

	parts := append([]string{}, prefix...)
	i, j := 0, 1
	for {
		if j < len(s) && !IsPathSeparator(s[j]) {
			// Keep looking for the end of this component.
			j++
			continue
		}
		parts = append(parts, s[i:j])
		// Advance to the next component, or end of the path.
		partEnd := j
		for j < len(s) && IsPathSeparator(s[j]) {
			j++
		}
		if j == len(s) {
			// If this is the last path component,
			// preserve any trailing path separators.
			suffixSep = s[partEnd:]
			break
		}
		if parts[len(parts)-1] == "." {
			// Remove "." components, except at the end.
			parts = parts[:len(parts)-1]
		}
		i = j
	}
	if len(suffix) > 0 && len(parts) > 0 && parts[len(parts)-1] == "." {
		// Remove a trailing "." component if we're joining to a suffix.
		parts = parts[:len(parts)-1]
	}
	parts = append(parts, suffix...)
	return parts, suffixSep, nil
}

// FS returns a file system (an fs.FS) for the tree of files in the root.
//
// The result implements [io/fs.StatFS], [io/fs.ReadFileFS] and
// [io/fs.ReadDirFS].
func (r *Root) FS() fs.FS {
	return (*rootFS)(r)
}

type rootFS Root

func (rfs *rootFS) Open(name string) (fs.File, error) {
	r := (*Root)(rfs)
	if !isValidRootFSPath(name) {
		return nil, &PathError{Op: "open", Path: name, Err: ErrInvalid}
	}
	f, err := r.Open(name)
	if err != nil {
		return nil, err
	}
	return f, nil
}

func (rfs *rootFS) ReadDir(name string) ([]DirEntry, error) {
	r := (*Root)(rfs)
	if !isValidRootFSPath(name) {
		return nil, &PathError{Op: "readdir", Path: name, Err: ErrInvalid}
	}

	// This isn't efficient: We just open a regular file and ReadDir it.
	// Ideally, we would skip creating a *File entirely and operate directly
	// on the file descriptor, but that will require some extensive reworking
	// of directory reading in general.
	//
	// This suffices for the moment.
	f, err := r.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dirs, err := f.ReadDir(-1)
	slices.SortFunc(dirs, func(a, b DirEntry) int {
		return bytealg.CompareString(a.Name(), b.Name())
	})
	return dirs, err
}

func (rfs *rootFS) ReadFile(name string) ([]byte, error) {
	r := (*Root)(rfs)
	if !isValidRootFSPath(name) {
		return nil, &PathError{Op: "readfile", Path: name, Err: ErrInvalid}
	}
	f, err := r.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readFileContents(f)
}

func (rfs *rootFS) Stat(name string) (FileInfo, error) {
	r := (*Root)(rfs)
	if !isValidRootFSPath(name) {
		return nil, &PathError{Op: "stat", Path: name, Err: ErrInvalid}
	}
	return r.Stat(name)
}

// isValidRootFSPath reprots whether name is a valid filename to pass a Root.FS method.
func isValidRootFSPath(name string) bool {
	if !fs.ValidPath(name) {
		return false
	}
	if runtime.GOOS == "windows" {
		// fs.FS paths are /-separated.
		// On Windows, reject the path if it contains any \ separators.
		// Other forms of invalid path (for example, "NUL") are handled by
		// Root's usual file lookup mechanisms.
		if stringslite.IndexByte(name, '\\') >= 0 {
			return false
		}
	}
	return true
}
