// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io/fs"
	"log"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// An Archive describes an archive to write: a collection of files.
// Directories are implied by the files and not explicitly listed.
type Archive struct {
	Files []File
}

// A File describes a single file to write to an archive.
type File struct {
	Name string    // name in archive
	Time time.Time // modification time
	Mode fs.FileMode
	Size int64
	Src  string // source file in OS file system
}

// Info returns a FileInfo about the file, for use with tar.FileInfoHeader
// and zip.FileInfoHeader.
func (f *File) Info() fs.FileInfo {
	return fileInfo{f}
}

// A fileInfo is an implementation of fs.FileInfo describing a File.
type fileInfo struct {
	f *File
}

func (i fileInfo) Name() string       { return path.Base(i.f.Name) }
func (i fileInfo) ModTime() time.Time { return i.f.Time }
func (i fileInfo) Mode() fs.FileMode  { return i.f.Mode }
func (i fileInfo) IsDir() bool        { return i.f.Mode&fs.ModeDir != 0 }
func (i fileInfo) Size() int64        { return i.f.Size }
func (i fileInfo) Sys() any           { return nil }

func (i fileInfo) String() string {
	return fs.FormatFileInfo(i)
}

// NewArchive returns a new Archive containing all the files in the directory dir.
// The archive can be amended afterward using methods like Add and Filter.
func NewArchive(dir string) (*Archive, error) {
	a := new(Archive)
	err := fs.WalkDir(os.DirFS(dir), ".", func(name string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		a.Add(name, filepath.Join(dir, name), info)
		return nil
	})
	if err != nil {
		return nil, err
	}
	a.Sort()
	return a, nil
}

// Add adds a file with the given name and info to the archive.
// The content of the file comes from the operating system file src.
// After a sequence of one or more calls to Add,
// the caller should invoke Sort to re-sort the archive's files.
func (a *Archive) Add(name, src string, info fs.FileInfo) {
	a.Files = append(a.Files, File{
		Name: name,
		Time: info.ModTime(),
		Mode: info.Mode(),
		Size: info.Size(),
		Src:  src,
	})
}

func nameLess(x, y string) bool {
	for i := 0; i < len(x) && i < len(y); i++ {
		if x[i] != y[i] {
			// foo/bar/baz before foo/bar.go, because foo/bar is before foo/bar.go
			if x[i] == '/' {
				return true
			}
			if y[i] == '/' {
				return false
			}
			return x[i] < y[i]
		}
	}
	return len(x) < len(y)
}

// Sort sorts the files in the archive.
// It is only necessary to call Sort after calling Add or RenameGoMod.
// NewArchive returns a sorted archive, and the other methods
// preserve the sorting of the archive.
func (a *Archive) Sort() {
	slices.SortFunc(a.Files, func(a, b File) int {
		if nameLess(a.Name, b.Name) {
			return -1
		} else if nameLess(b.Name, a.Name) {
			return +1
		} else {
			return 0
		}
	})
}

// Clone returns a copy of the Archive.
// Method calls like Add and Filter invoked on the copy do not affect the original,
// nor do calls on the original affect the copy.
func (a *Archive) Clone() *Archive {
	b := &Archive{
		Files: make([]File, len(a.Files)),
	}
	copy(b.Files, a.Files)
	return b
}

// AddPrefix adds a prefix to all file names in the archive.
func (a *Archive) AddPrefix(prefix string) {
	for i := range a.Files {
		a.Files[i].Name = path.Join(prefix, a.Files[i].Name)
	}
}

// Filter removes files from the archive for which keep(name) returns false.
func (a *Archive) Filter(keep func(name string) bool) {
	files := a.Files[:0]
	for _, f := range a.Files {
		if keep(f.Name) {
			files = append(files, f)
		}
	}
	a.Files = files
}

// SetMode changes the mode of every file in the archive
// to be mode(name, m), where m is the file's current mode.
func (a *Archive) SetMode(mode func(name string, m fs.FileMode) fs.FileMode) {
	for i := range a.Files {
		a.Files[i].Mode = mode(a.Files[i].Name, a.Files[i].Mode)
	}
}

// Remove removes files matching any of the patterns from the archive.
// The patterns use the syntax of path.Match, with an extension of allowing
// a leading **/ or trailing /**, which match any number of path elements
// (including no path elements) before or after the main match.
func (a *Archive) Remove(patterns ...string) {
	a.Filter(func(name string) bool {
		for _, pattern := range patterns {
			match, err := amatch(pattern, name)
			if err != nil {
				log.Fatalf("archive remove: %v", err)
			}
			if match {
				return false
			}
		}
		return true
	})
}

// SetTime sets the modification time of all files in the archive to t.
func (a *Archive) SetTime(t time.Time) {
	for i := range a.Files {
		a.Files[i].Time = t
	}
}

// RenameGoMod renames the go.mod files in the archive to _go.mod,
// for use with the module form, which cannot contain other go.mod files.
func (a *Archive) RenameGoMod() {
	for i, f := range a.Files {
		if strings.HasSuffix(f.Name, "/go.mod") {
			a.Files[i].Name = strings.TrimSuffix(f.Name, "go.mod") + "_go.mod"
		}
	}
}

func amatch(pattern, name string) (bool, error) {
	// firstN returns the prefix of name corresponding to the first n path elements.
	// If n <= 0, firstN returns the entire name.
	firstN := func(name string, n int) string {
		for i := 0; i < len(name); i++ {
			if name[i] == '/' {
				if n--; n == 0 {
					return name[:i]
				}
			}
		}
		return name
	}

	// lastN returns the suffix of name corresponding to the last n path elements.
	// If n <= 0, lastN returns the entire name.
	lastN := func(name string, n int) string {
		for i := len(name) - 1; i >= 0; i-- {
			if name[i] == '/' {
				if n--; n == 0 {
					return name[i+1:]
				}
			}
		}
		return name
	}

	if p, ok := strings.CutPrefix(pattern, "**/"); ok {
		return path.Match(p, lastN(name, 1+strings.Count(p, "/")))
	}
	if p, ok := strings.CutSuffix(pattern, "/**"); ok {
		return path.Match(p, firstN(name, 1+strings.Count(p, "/")))
	}
	return path.Match(pattern, name)
}
