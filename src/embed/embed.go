// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package embed provides access to files embedded in the running Go program.
//
// Go source files that import "embed" can use the //go:embed directive
// to initialize a variable of type string, []byte, or [FS] with the contents of
// files read from the package directory or subdirectories at compile time.
//
// For example, here are three ways to embed a file named hello.txt
// and then print its contents at run time.
//
// Embedding one file into a string:
//
//	import _ "embed"
//
//	//go:embed hello.txt
//	var s string
//	print(s)
//
// Embedding one file into a slice of bytes:
//
//	import _ "embed"
//
//	//go:embed hello.txt
//	var b []byte
//	print(string(b))
//
// Embedded one or more files into a file system:
//
//	import "embed"
//
//	//go:embed hello.txt
//	var f embed.FS
//	data, _ := f.ReadFile("hello.txt")
//	print(string(data))
//
// # Directives
//
// A //go:embed directive above a variable declaration specifies which files to embed,
// using one or more path.Match patterns.
//
// The directive must immediately precede a line containing the declaration of a single variable.
// Only blank lines and ‘//’ line comments are permitted between the directive and the declaration.
//
// The type of the variable must be a string type, or a slice of a byte type,
// or [FS] (or an alias of [FS]).
//
// For example:
//
//	package server
//
//	import "embed"
//
//	// content holds our static web server content.
//	//go:embed image/* template/*
//	//go:embed html/index.html
//	var content embed.FS
//
// The Go build system will recognize the directives and arrange for the declared variable
// (in the example above, content) to be populated with the matching files from the file system.
//
// The //go:embed directive accepts multiple space-separated patterns for
// brevity, but it can also be repeated, to avoid very long lines when there are
// many patterns. The patterns are interpreted relative to the package directory
// containing the source file. The path separator is a forward slash, even on
// Windows systems. Patterns may not contain ‘.’ or ‘..’ or empty path elements,
// nor may they begin or end with a slash. To match everything in the current
// directory, use ‘*’ instead of ‘.’. To allow for naming files with spaces in
// their names, patterns can be written as Go double-quoted or back-quoted
// string literals.
//
// If a pattern names a directory, all files in the subtree rooted at that directory are
// embedded (recursively), except that files with names beginning with ‘.’ or ‘_’
// are excluded. So the variable in the above example is almost equivalent to:
//
//	// content is our static web server content.
//	//go:embed image template html/index.html
//	var content embed.FS
//
// The difference is that ‘image/*’ embeds ‘image/.tempfile’ while ‘image’ does not.
// Neither embeds ‘image/dir/.tempfile’.
//
// If a pattern begins with the prefix ‘all:’, then the rule for walking directories is changed
// to include those files beginning with ‘.’ or ‘_’. For example, ‘all:image’ embeds
// both ‘image/.tempfile’ and ‘image/dir/.tempfile’.
//
// The //go:embed directive can be used with both exported and unexported variables,
// depending on whether the package wants to make the data available to other packages.
// It can only be used with variables at package scope, not with local variables.
//
// Patterns must not match files outside the package's module, such as ‘.git/*’, symbolic links,
// 'vendor/', or any directories containing go.mod (these are separate modules).
// Patterns must not match files whose names include the special punctuation characters  " * < > ? ` ' | / \ and :.
// Matches for empty directories are ignored. After that, each pattern in a //go:embed line
// must match at least one file or non-empty directory.
//
// If any patterns are invalid or have invalid matches, the build will fail.
//
// # Strings and Bytes
//
// The //go:embed line for a variable of type string or []byte can have only a single pattern,
// and that pattern can match only a single file. The string or []byte is initialized with
// the contents of that file.
//
// The //go:embed directive requires importing "embed", even when using a string or []byte.
// In source files that don't refer to [embed.FS], use a blank import (import _ "embed").
//
// # File Systems
//
// For embedding a single file, a variable of type string or []byte is often best.
// The [FS] type enables embedding a tree of files, such as a directory of static
// web server content, as in the example above.
//
// FS implements the [io/fs] package's [FS] interface, so it can be used with any package that
// understands file systems, including [net/http], [text/template], and [html/template].
//
// For example, given the content variable in the example above, we can write:
//
//	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.FS(content))))
//
//	template.ParseFS(content, "*.tmpl")
//
// # Tools
//
// To support tools that analyze Go packages, the patterns found in //go:embed lines
// are available in “go list” output. See the EmbedPatterns, TestEmbedPatterns,
// and XTestEmbedPatterns fields in the “go help list” output.
package embed

import (
	"errors"
	"internal/bytealg"
	"internal/stringslite"
	"io"
	"io/fs"
	"time"
)

// An FS is a read-only collection of files, usually initialized with a //go:embed directive.
// When declared without a //go:embed directive, an FS is an empty file system.
//
// An FS is a read-only value, so it is safe to use from multiple goroutines
// simultaneously and also safe to assign values of type FS to each other.
//
// FS implements fs.FS, so it can be used with any package that understands
// file system interfaces, including net/http, text/template, and html/template.
//
// See the package documentation for more details about initializing an FS.
type FS struct {
	// The compiler knows the layout of this struct.
	// See cmd/compile/internal/staticdata's WriteEmbed.
	//
	// The files list is sorted by name but not by simple string comparison.
	// Instead, each file's name takes the form "dir/elem" or "dir/elem/".
	// The optional trailing slash indicates that the file is itself a directory.
	// The files list is sorted first by dir (if dir is missing, it is taken to be ".")
	// and then by base, so this list of files:
	//
	//	p
	//	q/
	//	q/r
	//	q/s/
	//	q/s/t
	//	q/s/u
	//	q/v
	//	w
	//
	// is actually sorted as:
	//
	//	p       # dir=.    elem=p
	//	q/      # dir=.    elem=q
	//	w       # dir=.    elem=w
	//	q/r     # dir=q    elem=r
	//	q/s/    # dir=q    elem=s
	//	q/v     # dir=q    elem=v
	//	q/s/t   # dir=q/s  elem=t
	//	q/s/u   # dir=q/s  elem=u
	//
	// This order brings directory contents together in contiguous sections
	// of the list, allowing a directory read to use binary search to find
	// the relevant sequence of entries.
	files *[]file
}

// split splits the name into dir and elem as described in the
// comment in the FS struct above. isDir reports whether the
// final trailing slash was present, indicating that name is a directory.
func split(name string) (dir, elem string, isDir bool) {
	name, isDir = stringslite.CutSuffix(name, "/")
	i := bytealg.LastIndexByteString(name, '/')
	if i < 0 {
		return ".", name, isDir
	}
	return name[:i], name[i+1:], isDir
}

var (
	_ fs.ReadDirFS  = FS{}
	_ fs.ReadFileFS = FS{}
)

// A file is a single file in the FS.
// It implements fs.FileInfo and fs.DirEntry.
type file struct {
	// The compiler knows the layout of this struct.
	// See cmd/compile/internal/staticdata's WriteEmbed.
	name string
	data string
	hash [16]byte // truncated SHA256 hash
}

var (
	_ fs.FileInfo = (*file)(nil)
	_ fs.DirEntry = (*file)(nil)
)

func (f *file) Name() string               { _, elem, _ := split(f.name); return elem }
func (f *file) Size() int64                { return int64(len(f.data)) }
func (f *file) ModTime() time.Time         { return time.Time{} }
func (f *file) IsDir() bool                { _, _, isDir := split(f.name); return isDir }
func (f *file) Sys() any                   { return nil }
func (f *file) Type() fs.FileMode          { return f.Mode().Type() }
func (f *file) Info() (fs.FileInfo, error) { return f, nil }

func (f *file) Mode() fs.FileMode {
	if f.IsDir() {
		return fs.ModeDir | 0555
	}
	return 0444
}

func (f *file) String() string {
	return fs.FormatFileInfo(f)
}

// dotFile is a file for the root directory,
// which is omitted from the files list in a FS.
var dotFile = &file{name: "./"}

// lookup returns the named file, or nil if it is not present.
func (f FS) lookup(name string) *file {
	if !fs.ValidPath(name) {
		// The compiler should never emit a file with an invalid name,
		// so this check is not strictly necessary (if name is invalid,
		// we shouldn't find a match below), but it's a good backstop anyway.
		return nil
	}
	if name == "." {
		return dotFile
	}
	if f.files == nil {
		return nil
	}

	// Binary search to find where name would be in the list,
	// and then check if name is at that position.
	dir, elem, _ := split(name)
	files := *f.files
	i := sortSearch(len(files), func(i int) bool {
		idir, ielem, _ := split(files[i].name)
		return idir > dir || idir == dir && ielem >= elem
	})
	if i < len(files) && stringslite.TrimSuffix(files[i].name, "/") == name {
		return &files[i]
	}
	return nil
}

// readDir returns the list of files corresponding to the directory dir.
func (f FS) readDir(dir string) []file {
	if f.files == nil {
		return nil
	}
	// Binary search to find where dir starts and ends in the list
	// and then return that slice of the list.
	files := *f.files
	i := sortSearch(len(files), func(i int) bool {
		idir, _, _ := split(files[i].name)
		return idir >= dir
	})
	j := sortSearch(len(files), func(j int) bool {
		jdir, _, _ := split(files[j].name)
		return jdir > dir
	})
	return files[i:j]
}

// Open opens the named file for reading and returns it as an [fs.File].
//
// The returned file implements [io.Seeker] and [io.ReaderAt] when the file is not a directory.
func (f FS) Open(name string) (fs.File, error) {
	file := f.lookup(name)
	if file == nil {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}
	if file.IsDir() {
		return &openDir{file, f.readDir(name), 0}, nil
	}
	return &openFile{file, 0}, nil
}

// ReadDir reads and returns the entire named directory.
func (f FS) ReadDir(name string) ([]fs.DirEntry, error) {
	file, err := f.Open(name)
	if err != nil {
		return nil, err
	}
	dir, ok := file.(*openDir)
	if !ok {
		return nil, &fs.PathError{Op: "read", Path: name, Err: errors.New("not a directory")}
	}
	list := make([]fs.DirEntry, len(dir.files))
	for i := range list {
		list[i] = &dir.files[i]
	}
	return list, nil
}

// ReadFile reads and returns the content of the named file.
func (f FS) ReadFile(name string) ([]byte, error) {
	file, err := f.Open(name)
	if err != nil {
		return nil, err
	}
	ofile, ok := file.(*openFile)
	if !ok {
		return nil, &fs.PathError{Op: "read", Path: name, Err: errors.New("is a directory")}
	}
	return []byte(ofile.f.data), nil
}

// An openFile is a regular file open for reading.
type openFile struct {
	f      *file // the file itself
	offset int64 // current read offset
}

var (
	_ io.Seeker   = (*openFile)(nil)
	_ io.ReaderAt = (*openFile)(nil)
)

func (f *openFile) Close() error               { return nil }
func (f *openFile) Stat() (fs.FileInfo, error) { return f.f, nil }

func (f *openFile) Read(b []byte) (int, error) {
	if f.offset >= int64(len(f.f.data)) {
		return 0, io.EOF
	}
	if f.offset < 0 {
		return 0, &fs.PathError{Op: "read", Path: f.f.name, Err: fs.ErrInvalid}
	}
	n := copy(b, f.f.data[f.offset:])
	f.offset += int64(n)
	return n, nil
}

func (f *openFile) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case 0:
		// offset += 0
	case 1:
		offset += f.offset
	case 2:
		offset += int64(len(f.f.data))
	}
	if offset < 0 || offset > int64(len(f.f.data)) {
		return 0, &fs.PathError{Op: "seek", Path: f.f.name, Err: fs.ErrInvalid}
	}
	f.offset = offset
	return offset, nil
}

func (f *openFile) ReadAt(b []byte, offset int64) (int, error) {
	if offset < 0 || offset > int64(len(f.f.data)) {
		return 0, &fs.PathError{Op: "read", Path: f.f.name, Err: fs.ErrInvalid}
	}
	n := copy(b, f.f.data[offset:])
	if n < len(b) {
		return n, io.EOF
	}
	return n, nil
}

// An openDir is a directory open for reading.
type openDir struct {
	f      *file  // the directory file itself
	files  []file // the directory contents
	offset int    // the read offset, an index into the files slice
}

func (d *openDir) Close() error               { return nil }
func (d *openDir) Stat() (fs.FileInfo, error) { return d.f, nil }

func (d *openDir) Read([]byte) (int, error) {
	return 0, &fs.PathError{Op: "read", Path: d.f.name, Err: errors.New("is a directory")}
}

func (d *openDir) ReadDir(count int) ([]fs.DirEntry, error) {
	n := len(d.files) - d.offset
	if n == 0 {
		if count <= 0 {
			return nil, nil
		}
		return nil, io.EOF
	}
	if count > 0 && n > count {
		n = count
	}
	list := make([]fs.DirEntry, n)
	for i := range list {
		list[i] = &d.files[d.offset+i]
	}
	d.offset += n
	return list, nil
}

// sortSearch is like sort.Search, avoiding an import.
func sortSearch(n int, f func(int) bool) int {
	// Define f(-1) == false and f(n) == true.
	// Invariant: f(i-1) == false, f(j) == true.
	i, j := 0, n
	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if !f(h) {
			i = h + 1 // preserves f(i-1) == false
		} else {
			j = h // preserves f(j) == true
		}
	}
	// i == j, f(i-1) == false, and f(j) (= f(i)) == true  =>  answer is i.
	return i
}
