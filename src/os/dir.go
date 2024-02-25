// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/safefilepath"
	"io"
	"io/fs"
	"sort"
)

type readdirMode int

const (
	readdirName readdirMode = iota
	readdirDirEntry
	readdirFileInfo
)

// Readdir reads the contents of the directory associated with file and
// returns a slice of up to n FileInfo values, as would be returned
// by Lstat, in directory order. Subsequent calls on the same file will yield
// further FileInfos.
//
// If n > 0, Readdir returns at most n FileInfo structures. In this case, if
// Readdir returns an empty slice, it will return a non-nil error
// explaining why. At the end of a directory, the error is io.EOF.
//
// If n <= 0, Readdir returns all the FileInfo from the directory in
// a single slice. In this case, if Readdir succeeds (reads all
// the way to the end of the directory), it returns the slice and a
// nil error. If it encounters an error before the end of the
// directory, Readdir returns the FileInfo read until that point
// and a non-nil error.
//
// Most clients are better served by the more efficient ReadDir method.
func (f *File) Readdir(n int) ([]FileInfo, error) {
	if f == nil {
		return nil, ErrInvalid
	}
	_, _, infos, err := f.readdir(n, readdirFileInfo)
	if infos == nil {
		// Readdir has historically always returned a non-nil empty slice, never nil,
		// even on error (except misuse with nil receiver above).
		// Keep it that way to avoid breaking overly sensitive callers.
		infos = []FileInfo{}
	}
	return infos, err
}

// Readdirnames reads the contents of the directory associated with file
// and returns a slice of up to n names of files in the directory,
// in directory order. Subsequent calls on the same file will yield
// further names.
//
// If n > 0, Readdirnames returns at most n names. In this case, if
// Readdirnames returns an empty slice, it will return a non-nil error
// explaining why. At the end of a directory, the error is io.EOF.
//
// If n <= 0, Readdirnames returns all the names from the directory in
// a single slice. In this case, if Readdirnames succeeds (reads all
// the way to the end of the directory), it returns the slice and a
// nil error. If it encounters an error before the end of the
// directory, Readdirnames returns the names read until that point and
// a non-nil error.
func (f *File) Readdirnames(n int) (names []string, err error) {
	if f == nil {
		return nil, ErrInvalid
	}
	names, _, _, err = f.readdir(n, readdirName)
	if names == nil {
		// Readdirnames has historically always returned a non-nil empty slice, never nil,
		// even on error (except misuse with nil receiver above).
		// Keep it that way to avoid breaking overly sensitive callers.
		names = []string{}
	}
	return names, err
}

// A DirEntry is an entry read from a directory
// (using the ReadDir function or a File's ReadDir method).
type DirEntry = fs.DirEntry

// ReadDir reads the contents of the directory associated with the file f
// and returns a slice of DirEntry values in directory order.
// Subsequent calls on the same file will yield later DirEntry records in the directory.
//
// If n > 0, ReadDir returns at most n DirEntry records.
// In this case, if ReadDir returns an empty slice, it will return an error explaining why.
// At the end of a directory, the error is io.EOF.
//
// If n <= 0, ReadDir returns all the DirEntry records remaining in the directory.
// When it succeeds, it returns a nil error (not io.EOF).
func (f *File) ReadDir(n int) ([]DirEntry, error) {
	if f == nil {
		return nil, ErrInvalid
	}
	_, dirents, _, err := f.readdir(n, readdirDirEntry)
	if dirents == nil {
		// Match Readdir and Readdirnames: don't return nil slices.
		dirents = []DirEntry{}
	}
	return dirents, err
}

// testingForceReadDirLstat forces ReadDir to call Lstat, for testing that code path.
// This can be difficult to provoke on some Unix systems otherwise.
var testingForceReadDirLstat bool

// ReadDir reads the named directory,
// returning all its directory entries sorted by filename.
// If an error occurs reading the directory,
// ReadDir returns the entries it was able to read before the error,
// along with the error.
func ReadDir(name string) ([]DirEntry, error) {
	f, err := Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	dirs, err := f.ReadDir(-1)
	sort.Slice(dirs, func(i, j int) bool { return dirs[i].Name() < dirs[j].Name() })
	return dirs, err
}

// CopyFS copies the file system fsys into the directory dir,
// creating dir if necessary.
//
// Newly created directories and files have their default modes
// where any bits from the file in fsys that are not part of the
// standard read, write, and execute permissions will be zeroed
// out, and standard read and write permissions are set for owner,
// group, and others while retaining any existing execute bits from
// the file in fsys.
//
// Symbolic links in fsys are not supported, a *PathError with Err set
// to ErrInvalid is returned on symlink.
//
// Copying stops at and returns the first error encountered.
func CopyFS(dir string, fsys fs.FS) error {
	return fs.WalkDir(fsys, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		fpath, err := safefilepath.FromFS(path)
		if err != nil {
			return err
		}
		newPath := joinPath(dir, fpath)
		if d.IsDir() {
			return MkdirAll(newPath, 0777)
		}

		// TODO(panjf2000): handle symlinks with the help of fs.ReadLinkFS
		// 		once https://go.dev/issue/49580 is done.
		//		we also need safefilepath.IsLocal from https://go.dev/cl/564295.
		if !d.Type().IsRegular() {
			return &PathError{Op: "CopyFS", Path: path, Err: ErrInvalid}
		}

		r, err := fsys.Open(path)
		if err != nil {
			return err
		}
		defer r.Close()
		info, err := r.Stat()
		if err != nil {
			return err
		}
		w, err := OpenFile(newPath, O_CREATE|O_TRUNC|O_WRONLY, 0666|info.Mode()&0777)
		if err != nil {
			return err
		}

		if _, err := io.Copy(w, r); err != nil {
			w.Close()
			return &PathError{Op: "Copy", Path: newPath, Err: err}
		}
		return w.Close()
	})
}
