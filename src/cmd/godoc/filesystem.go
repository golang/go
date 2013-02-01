// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines types for abstract file system access and
// provides an implementation accessing the file system of the
// underlying OS.

package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	pathpkg "path"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// fs is the file system that godoc reads from and serves.
// It is a virtual file system that operates on slash-separated paths,
// and its root corresponds to the Go distribution root: /src/pkg
// holds the source tree, and so on.  This means that the URLs served by
// the godoc server are the same as the paths in the virtual file
// system, which helps keep things simple.
//
// New file trees - implementations of FileSystem - can be added to
// the virtual file system using nameSpace's Bind method.
// The usual setup is to bind OS(runtime.GOROOT) to the root
// of the name space and then bind any GOPATH/src directories
// on top of /src/pkg, so that all sources are in /src/pkg.
//
// For more about name spaces, see the nameSpace type's
// documentation below.
//
// The use of this virtual file system means that most code processing
// paths can assume they are slash-separated and should be using
// package path (often imported as pathpkg) to manipulate them,
// even on Windows.
//
var fs = nameSpace{} // the underlying file system for godoc

// Setting debugNS = true will enable debugging prints about
// name space translations.
const debugNS = false

// The FileSystem interface specifies the methods godoc is using
// to access the file system for which it serves documentation.
type FileSystem interface {
	Open(path string) (readSeekCloser, error)
	Lstat(path string) (os.FileInfo, error)
	Stat(path string) (os.FileInfo, error)
	ReadDir(path string) ([]os.FileInfo, error)
	String() string
}

type readSeekCloser interface {
	io.Reader
	io.Seeker
	io.Closer
}

// ReadFile reads the file named by path from fs and returns the contents.
func ReadFile(fs FileSystem, path string) ([]byte, error) {
	rc, err := fs.Open(path)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return ioutil.ReadAll(rc)
}

// OS returns an implementation of FileSystem reading from the
// tree rooted at root.  Recording a root is convenient everywhere
// but necessary on Windows, because the slash-separated path
// passed to Open has no way to specify a drive letter.  Using a root
// lets code refer to OS(`c:\`), OS(`d:\`) and so on.
func OS(root string) FileSystem {
	return osFS(root)
}

type osFS string

func (root osFS) String() string { return "os(" + string(root) + ")" }

func (root osFS) resolve(path string) string {
	// Clean the path so that it cannot possibly begin with ../.
	// If it did, the result of filepath.Join would be outside the
	// tree rooted at root.  We probably won't ever see a path
	// with .. in it, but be safe anyway.
	path = pathpkg.Clean("/" + path)

	return filepath.Join(string(root), path)
}

func (root osFS) Open(path string) (readSeekCloser, error) {
	f, err := os.Open(root.resolve(path))
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return nil, fmt.Errorf("Open: %s is a directory", path)
	}
	return f, nil
}

func (root osFS) Lstat(path string) (os.FileInfo, error) {
	return os.Lstat(root.resolve(path))
}

func (root osFS) Stat(path string) (os.FileInfo, error) {
	return os.Stat(root.resolve(path))
}

func (root osFS) ReadDir(path string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(root.resolve(path)) // is sorted
}

// hasPathPrefix returns true if x == y or x == y + "/" + more
func hasPathPrefix(x, y string) bool {
	return x == y || strings.HasPrefix(x, y) && (strings.HasSuffix(y, "/") || strings.HasPrefix(x[len(y):], "/"))
}

// A nameSpace is a file system made up of other file systems
// mounted at specific locations in the name space.
//
// The representation is a map from mount point locations
// to the list of file systems mounted at that location.  A traditional
// Unix mount table would use a single file system per mount point,
// but we want to be able to mount multiple file systems on a single
// mount point and have the system behave as if the union of those
// file systems were present at the mount point.
// For example, if the OS file system has a Go installation in
// c:\Go and additional Go path trees in  d:\Work1 and d:\Work2, then
// this name space creates the view we want for the godoc server:
//
//	nameSpace{
//		"/": {
//			{old: "/", fs: OS(`c:\Go`), new: "/"},
//		},
//		"/src/pkg": {
//			{old: "/src/pkg", fs: OS(`c:\Go`), new: "/src/pkg"},
//			{old: "/src/pkg", fs: OS(`d:\Work1`), new: "/src"},
//			{old: "/src/pkg", fs: OS(`d:\Work2`), new: "/src"},
//		},
//	}
//
// This is created by executing:
//
//	ns := nameSpace{}
//	ns.Bind("/", OS(`c:\Go`), "/", bindReplace)
//	ns.Bind("/src/pkg", OS(`d:\Work1`), "/src", bindAfter)
//	ns.Bind("/src/pkg", OS(`d:\Work2`), "/src", bindAfter)
//
// A particular mount point entry is a triple (old, fs, new), meaning that to
// operate on a path beginning with old, replace that prefix (old) with new
// and then pass that path to the FileSystem implementation fs.
//
// Given this name space, a ReadDir of /src/pkg/code will check each prefix
// of the path for a mount point (first /src/pkg/code, then /src/pkg, then /src,
// then /), stopping when it finds one.  For the above example, /src/pkg/code
// will find the mount point at /src/pkg:
//
//	{old: "/src/pkg", fs: OS(`c:\Go`), new: "/src/pkg"},
//	{old: "/src/pkg", fs: OS(`d:\Work1`), new: "/src"},
//	{old: "/src/pkg", fs: OS(`d:\Work2`), new: "/src"},
//
// ReadDir will when execute these three calls and merge the results:
//
//	OS(`c:\Go`).ReadDir("/src/pkg/code")
//	OS(`d:\Work1').ReadDir("/src/code")
//	OS(`d:\Work2').ReadDir("/src/code")
//
// Note that the "/src/pkg" in "/src/pkg/code" has been replaced by
// just "/src" in the final two calls.
//
// OS is itself an implementation of a file system: it implements
// OS(`c:\Go`).ReadDir("/src/pkg/code") as ioutil.ReadDir(`c:\Go\src\pkg\code`).
//
// Because the new path is evaluated by fs (here OS(root)), another way
// to read the mount table is to mentally combine fs+new, so that this table:
//
//	{old: "/src/pkg", fs: OS(`c:\Go`), new: "/src/pkg"},
//	{old: "/src/pkg", fs: OS(`d:\Work1`), new: "/src"},
//	{old: "/src/pkg", fs: OS(`d:\Work2`), new: "/src"},
//
// reads as:
//
//	"/src/pkg" -> c:\Go\src\pkg
//	"/src/pkg" -> d:\Work1\src
//	"/src/pkg" -> d:\Work2\src
//
// An invariant (a redundancy) of the name space representation is that
// ns[mtpt][i].old is always equal to mtpt (in the example, ns["/src/pkg"]'s
// mount table entries always have old == "/src/pkg").  The 'old' field is
// useful to callers, because they receive just a []mountedFS and not any
// other indication of which mount point was found.
//
type nameSpace map[string][]mountedFS

// A mountedFS handles requests for path by replacing
// a prefix 'old' with 'new' and then calling the fs methods.
type mountedFS struct {
	old string
	fs  FileSystem
	new string
}

// translate translates path for use in m, replacing old with new.
//
// mountedFS{"/src/pkg", fs, "/src"}.translate("/src/pkg/code") == "/src/code".
func (m mountedFS) translate(path string) string {
	path = pathpkg.Clean("/" + path)
	if !hasPathPrefix(path, m.old) {
		panic("translate " + path + " but old=" + m.old)
	}
	return pathpkg.Join(m.new, path[len(m.old):])
}

func (nameSpace) String() string {
	return "ns"
}

// Fprint writes a text representation of the name space to w.
func (ns nameSpace) Fprint(w io.Writer) {
	fmt.Fprint(w, "name space {\n")
	var all []string
	for mtpt := range ns {
		all = append(all, mtpt)
	}
	sort.Strings(all)
	for _, mtpt := range all {
		fmt.Fprintf(w, "\t%s:\n", mtpt)
		for _, m := range ns[mtpt] {
			fmt.Fprintf(w, "\t\t%s %s\n", m.fs, m.new)
		}
	}
	fmt.Fprint(w, "}\n")
}

// clean returns a cleaned, rooted path for evaluation.
// It canonicalizes the path so that we can use string operations
// to analyze it.
func (nameSpace) clean(path string) string {
	return pathpkg.Clean("/" + path)
}

// Bind causes references to old to redirect to the path new in newfs.
// If mode is bindReplace, old redirections are discarded.
// If mode is bindBefore, this redirection takes priority over existing ones,
// but earlier ones are still consulted for paths that do not exist in newfs.
// If mode is bindAfter, this redirection happens only after existing ones
// have been tried and failed.

const (
	bindReplace = iota
	bindBefore
	bindAfter
)

func (ns nameSpace) Bind(old string, newfs FileSystem, new string, mode int) {
	old = ns.clean(old)
	new = ns.clean(new)
	m := mountedFS{old, newfs, new}
	var mtpt []mountedFS
	switch mode {
	case bindReplace:
		mtpt = append(mtpt, m)
	case bindAfter:
		mtpt = append(mtpt, ns.resolve(old)...)
		mtpt = append(mtpt, m)
	case bindBefore:
		mtpt = append(mtpt, m)
		mtpt = append(mtpt, ns.resolve(old)...)
	}

	// Extend m.old, m.new in inherited mount point entries.
	for i := range mtpt {
		m := &mtpt[i]
		if m.old != old {
			if !hasPathPrefix(old, m.old) {
				// This should not happen.  If it does, panic so
				// that we can see the call trace that led to it.
				panic(fmt.Sprintf("invalid Bind: old=%q m={%q, %s, %q}", old, m.old, m.fs.String(), m.new))
			}
			suffix := old[len(m.old):]
			m.old = pathpkg.Join(m.old, suffix)
			m.new = pathpkg.Join(m.new, suffix)
		}
	}

	ns[old] = mtpt
}

// resolve resolves a path to the list of mountedFS to use for path.
func (ns nameSpace) resolve(path string) []mountedFS {
	path = ns.clean(path)
	for {
		if m := ns[path]; m != nil {
			if debugNS {
				fmt.Printf("resolve %s: %v\n", path, m)
			}
			return m
		}
		if path == "/" {
			break
		}
		path = pathpkg.Dir(path)
	}
	return nil
}

// Open implements the FileSystem Open method.
func (ns nameSpace) Open(path string) (readSeekCloser, error) {
	var err error
	for _, m := range ns.resolve(path) {
		if debugNS {
			fmt.Printf("tx %s: %v\n", path, m.translate(path))
		}
		r, err1 := m.fs.Open(m.translate(path))
		if err1 == nil {
			return r, nil
		}
		if err == nil {
			err = err1
		}
	}
	if err == nil {
		err = &os.PathError{Op: "open", Path: path, Err: os.ErrNotExist}
	}
	return nil, err
}

// stat implements the FileSystem Stat and Lstat methods.
func (ns nameSpace) stat(path string, f func(FileSystem, string) (os.FileInfo, error)) (os.FileInfo, error) {
	var err error
	for _, m := range ns.resolve(path) {
		fi, err1 := f(m.fs, m.translate(path))
		if err1 == nil {
			return fi, nil
		}
		if err == nil {
			err = err1
		}
	}
	if err == nil {
		err = &os.PathError{Op: "stat", Path: path, Err: os.ErrNotExist}
	}
	return nil, err
}

func (ns nameSpace) Stat(path string) (os.FileInfo, error) {
	return ns.stat(path, FileSystem.Stat)
}

func (ns nameSpace) Lstat(path string) (os.FileInfo, error) {
	return ns.stat(path, FileSystem.Lstat)
}

// dirInfo is a trivial implementation of os.FileInfo for a directory.
type dirInfo string

func (d dirInfo) Name() string       { return string(d) }
func (d dirInfo) Size() int64        { return 0 }
func (d dirInfo) Mode() os.FileMode  { return os.ModeDir | 0555 }
func (d dirInfo) ModTime() time.Time { return startTime }
func (d dirInfo) IsDir() bool        { return true }
func (d dirInfo) Sys() interface{}   { return nil }

var startTime = time.Now()

// ReadDir implements the FileSystem ReadDir method.  It's where most of the magic is.
// (The rest is in resolve.)
//
// Logically, ReadDir must return the union of all the directories that are named
// by path.  In order to avoid misinterpreting Go packages, of all the directories
// that contain Go source code, we only include the files from the first,
// but we include subdirectories from all.
//
// ReadDir must also return directory entries needed to reach mount points.
// If the name space looks like the example in the type nameSpace comment,
// but c:\Go does not have a src/pkg subdirectory, we still want to be able
// to find that subdirectory, because we've mounted d:\Work1 and d:\Work2
// there.  So if we don't see "src" in the directory listing for c:\Go, we add an
// entry for it before returning.
//
func (ns nameSpace) ReadDir(path string) ([]os.FileInfo, error) {
	path = ns.clean(path)

	var (
		haveGo   = false
		haveName = map[string]bool{}
		all      []os.FileInfo
		err      error
		first    []os.FileInfo
	)

	for _, m := range ns.resolve(path) {
		dir, err1 := m.fs.ReadDir(m.translate(path))
		if err1 != nil {
			if err == nil {
				err = err1
			}
			continue
		}

		if dir == nil {
			dir = []os.FileInfo{}
		}

		if first == nil {
			first = dir
		}

		// If we don't yet have Go files in 'all' and this directory
		// has some, add all the files from this directory.
		// Otherwise, only add subdirectories.
		useFiles := false
		if !haveGo {
			for _, d := range dir {
				if strings.HasSuffix(d.Name(), ".go") {
					useFiles = true
					haveGo = true
					break
				}
			}
		}

		for _, d := range dir {
			name := d.Name()
			if (d.IsDir() || useFiles) && !haveName[name] {
				haveName[name] = true
				all = append(all, d)
			}
		}
	}

	// We didn't find any directories containing Go files.
	// If some directory returned successfully, use that.
	if !haveGo {
		for _, d := range first {
			if !haveName[d.Name()] {
				haveName[d.Name()] = true
				all = append(all, d)
			}
		}
	}

	// Built union.  Add any missing directories needed to reach mount points.
	for old := range ns {
		if hasPathPrefix(old, path) && old != path {
			// Find next element after path in old.
			elem := old[len(path):]
			elem = strings.TrimPrefix(elem, "/")
			if i := strings.Index(elem, "/"); i >= 0 {
				elem = elem[:i]
			}
			if !haveName[elem] {
				haveName[elem] = true
				all = append(all, dirInfo(elem))
			}
		}
	}

	if len(all) == 0 {
		return nil, err
	}

	sort.Sort(byName(all))
	return all, nil
}

// byName implements sort.Interface.
type byName []os.FileInfo

func (f byName) Len() int           { return len(f) }
func (f byName) Less(i, j int) bool { return f[i].Name() < f[j].Name() }
func (f byName) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }

// An httpFS implements http.FileSystem using a FileSystem.
type httpFS struct {
	fs FileSystem
}

func (h *httpFS) Open(name string) (http.File, error) {
	fi, err := h.fs.Stat(name)
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return &httpDir{h.fs, name, nil}, nil
	}
	f, err := h.fs.Open(name)
	if err != nil {
		return nil, err
	}
	return &httpFile{h.fs, f, name}, nil
}

// httpDir implements http.File for a directory in a FileSystem.
type httpDir struct {
	fs      FileSystem
	name    string
	pending []os.FileInfo
}

func (h *httpDir) Close() error               { return nil }
func (h *httpDir) Stat() (os.FileInfo, error) { return h.fs.Stat(h.name) }
func (h *httpDir) Read([]byte) (int, error) {
	return 0, fmt.Errorf("cannot Read from directory %s", h.name)
}

func (h *httpDir) Seek(offset int64, whence int) (int64, error) {
	if offset == 0 && whence == 0 {
		h.pending = nil
		return 0, nil
	}
	return 0, fmt.Errorf("unsupported Seek in directory %s", h.name)
}

func (h *httpDir) Readdir(count int) ([]os.FileInfo, error) {
	if h.pending == nil {
		d, err := h.fs.ReadDir(h.name)
		if err != nil {
			return nil, err
		}
		if d == nil {
			d = []os.FileInfo{} // not nil
		}
		h.pending = d
	}

	if len(h.pending) == 0 && count > 0 {
		return nil, io.EOF
	}
	if count <= 0 || count > len(h.pending) {
		count = len(h.pending)
	}
	d := h.pending[:count]
	h.pending = h.pending[count:]
	return d, nil
}

// httpFile implements http.File for a file (not directory) in a FileSystem.
type httpFile struct {
	fs FileSystem
	readSeekCloser
	name string
}

func (h *httpFile) Stat() (os.FileInfo, error) { return h.fs.Stat(h.name) }
func (h *httpFile) Readdir(int) ([]os.FileInfo, error) {
	return nil, fmt.Errorf("cannot Readdir from file %s", h.name)
}
