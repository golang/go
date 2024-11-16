// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fsys implements a virtual file system that the go command
// uses to read source file trees. The virtual file system redirects some
// OS file paths to other OS file paths, according to an overlay file.
// Editors can use this overlay support to invoke the go command on
// temporary files that have been edited but not yet saved into their
// final locations.
package fsys

import (
	"encoding/json"
	"errors"
	"fmt"
	"internal/godebug"
	"io"
	"io/fs"
	"iter"
	"log"
	"maps"
	"os"
	pathpkg "path"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"
)

// Trace emits a trace event for the operation and file path to the trace log,
// but only when $GODEBUG contains gofsystrace=1.
// The traces are appended to the file named by the $GODEBUG setting gofsystracelog, or else standard error.
// For debugging, if the $GODEBUG setting gofsystracestack is non-empty, then trace events for paths
// matching that glob pattern (using path.Match) will be followed by a full stack trace.
func Trace(op, path string) {
	if !doTrace {
		return
	}
	traceMu.Lock()
	defer traceMu.Unlock()
	fmt.Fprintf(traceFile, "%d gofsystrace %s %s\n", os.Getpid(), op, path)
	if pattern := gofsystracestack.Value(); pattern != "" {
		if match, _ := pathpkg.Match(pattern, path); match {
			traceFile.Write(debug.Stack())
		}
	}
}

var (
	doTrace   bool
	traceFile *os.File
	traceMu   sync.Mutex

	gofsystrace      = godebug.New("#gofsystrace")
	gofsystracelog   = godebug.New("#gofsystracelog")
	gofsystracestack = godebug.New("#gofsystracestack")
)

func init() {
	if gofsystrace.Value() != "1" {
		return
	}
	doTrace = true
	if f := gofsystracelog.Value(); f != "" {
		// Note: No buffering on writes to this file, so no need to worry about closing it at exit.
		var err error
		traceFile, err = os.OpenFile(f, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0666)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		traceFile = os.Stderr
	}
}

// OverlayFile is the -overlay flag value.
// It names a file containing the JSON for an overlayJSON struct.
var OverlayFile string

// overlayJSON is the format for the -overlay file.
type overlayJSON struct {
	// Replace maps file names observed by Go tools
	// to the actual files that should be used when those are read.
	// If the actual name is "", the file should appear to be deleted.
	Replace map[string]string
}

type replace struct {
	from string
	to   string
}

type node struct {
	actual   string           // empty if a directory
	children map[string]*node // path element → file or directory
}

func (n *node) isDir() bool {
	return n.actual == "" && n.children != nil
}

func (n *node) isDeleted() bool {
	return n.actual == "" && n.children == nil
}

// TODO(matloob): encapsulate these in an io/fs-like interface
var overlay map[string]*node // path -> file or directory node

// cwd returns the current directory, caching it on first use.
var cwd = sync.OnceValue(cwdOnce)

func cwdOnce() string {
	wd, err := os.Getwd()
	if err != nil {
		// Note: cannot import base, so using log.Fatal.
		log.Fatalf("cannot determine current directory: %v", err)
	}
	return wd
}

// abs returns the absolute form of path, for looking up in the overlay map.
// For the most part, this is filepath.Abs and filepath.Clean,
// except that Windows requires special handling, as always.
func abs(path string) string {
	if path == "" {
		return ""
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}

	dir := cwd()
	if vol := filepath.VolumeName(dir); vol != "" && (path[0] == '\\' || path[0] == '/') {
		// path is volume-relative, like `\Temp`.
		// Connect to volume name to make absolute path.
		// See go.dev/issue/8130.
		return filepath.Join(vol, path)
	}

	return filepath.Join(dir, path)
}

// info is a summary of the known information about a path
// being looked up in the virtual file system.
type info struct {
	abs      string
	deleted  bool
	replaced bool
	dir      bool
	actual   string
}

// stat returns info about the path in the virtual file system.
func stat(path string) info {
	apath := abs(path)
	if n, ok := overlay[apath]; ok {
		if n.isDir() {
			return info{abs: apath, replaced: true, dir: true, actual: path}
		}
		if n.isDeleted() {
			return info{abs: apath, deleted: true}
		}
		return info{abs: apath, replaced: true, actual: n.actual}
	}

	// Check whether any parents are replaced by files,
	// meaning this path and the directory that contained it
	// have been deleted.
	prefix := apath
	for {
		if n, ok := overlay[prefix]; ok {
			if n.children == nil {
				return info{abs: apath, deleted: true}
			}
			break
		}
		parent := filepath.Dir(prefix)
		if parent == prefix {
			break
		}
		prefix = parent
	}

	return info{abs: apath, actual: path}
}

// children returns a sequence of (name, info)
// for all the children of the directory i.
func (i *info) children() iter.Seq2[string, info] {
	return func(yield func(string, info) bool) {
		n := overlay[i.abs]
		if n == nil {
			return
		}
		for name, c := range n.children {
			ci := info{
				abs:      filepath.Join(i.abs, name),
				deleted:  c.isDeleted(),
				replaced: c.children != nil || c.actual != "",
				dir:      c.isDir(),
				actual:   c.actual,
			}
			if !yield(name, ci) {
				return
			}
		}
	}
}

// Init initializes the overlay, if one is being used.
func Init() error {
	if overlay != nil {
		// already initialized
		return nil
	}

	if OverlayFile == "" {
		return nil
	}

	Trace("ReadFile", OverlayFile)
	b, err := os.ReadFile(OverlayFile)
	if err != nil {
		return fmt.Errorf("reading overlay: %v", err)
	}
	return initFromJSON(b)
}

func initFromJSON(js []byte) error {
	var ojs overlayJSON
	if err := json.Unmarshal(js, &ojs); err != nil {
		return fmt.Errorf("parsing overlay JSON: %v", err)
	}

	seen := make(map[string]string)
	var list []replace
	for _, from := range slices.Sorted(maps.Keys(ojs.Replace)) {
		if from == "" {
			return fmt.Errorf("empty string key in overlay map")
		}
		afrom := abs(from)
		if old, ok := seen[afrom]; ok {
			return fmt.Errorf("duplicate paths %s and %s in overlay map", old, from)
		}
		seen[afrom] = from
		list = append(list, replace{from: afrom, to: ojs.Replace[from]})
	}

	slices.SortFunc(list, func(x, y replace) int { return cmp(x.from, y.from) })

	for i, r := range list {
		if r.to == "" { // deleted
			continue
		}
		// have file for r.from; look for child file implying r.from is a directory
		prefix := r.from + string(filepath.Separator)
		for _, next := range list[i+1:] {
			if !strings.HasPrefix(next.from, prefix) {
				break
			}
			if next.to != "" {
				// found child file
				return fmt.Errorf("inconsistent files %s and %s in overlay map", r.from, next.from)
			}
		}
	}

	overlay = make(map[string]*node)
	for _, r := range list {
		n := &node{actual: abs(r.to)}
		from := r.from
		overlay[from] = n

		for {
			dir, base := filepath.Dir(from), filepath.Base(from)
			if dir == from {
				break
			}
			dn := overlay[dir]
			if dn == nil || dn.isDeleted() {
				dn = &node{children: make(map[string]*node)}
				overlay[dir] = dn
			}
			if n.isDeleted() && !dn.isDir() {
				break
			}
			if !dn.isDir() {
				panic("fsys inconsistency")
			}
			dn.children[base] = n
			if n.isDeleted() {
				// Deletion is recorded now.
				// Don't need to create entire parent chain,
				// because we don't need to force parents to exist.
				break
			}
			from, n = dir, dn
		}
	}

	return nil
}

// IsDir returns true if path is a directory on disk or in the
// overlay.
func IsDir(path string) (bool, error) {
	Trace("IsDir", path)

	switch info := stat(path); {
	case info.dir:
		return true, nil
	case info.deleted, info.replaced:
		return false, nil
	}

	info, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	return info.IsDir(), nil
}

// errNotDir is used to communicate from ReadDir to IsGoDir
// that the argument is not a directory, so that IsGoDir doesn't
// return an error.
var errNotDir = errors.New("not a directory")

// osReadDir is like os.ReadDir corrects the error to be errNotDir
// if the problem is that name exists but is not a directory.
func osReadDir(name string) ([]fs.DirEntry, error) {
	dirs, err := os.ReadDir(name)
	if err != nil && !os.IsNotExist(err) {
		if info, err := os.Stat(name); err == nil && !info.IsDir() {
			return nil, &fs.PathError{Op: "ReadDir", Path: name, Err: errNotDir}
		}
	}
	return dirs, err
}

// ReadDir reads the named directory in the virtual file system.
func ReadDir(name string) ([]fs.DirEntry, error) {
	Trace("ReadDir", name)

	info := stat(name)
	if info.deleted {
		return nil, &fs.PathError{Op: "read", Path: name, Err: fs.ErrNotExist}
	}
	if !info.replaced {
		return osReadDir(name)
	}
	if !info.dir {
		return nil, &fs.PathError{Op: "read", Path: name, Err: errNotDir}
	}

	// Start with normal disk listing.
	dirs, err := osReadDir(name)
	if err != nil && !os.IsNotExist(err) && !errors.Is(err, errNotDir) {
		return nil, err
	}

	// Merge disk listing and overlay entries in map.
	all := make(map[string]fs.DirEntry)
	for _, d := range dirs {
		all[d.Name()] = d
	}
	for cname, cinfo := range info.children() {
		if cinfo.dir {
			all[cname] = fs.FileInfoToDirEntry(fakeDir(cname))
			continue
		}
		if cinfo.deleted {
			delete(all, cname)
			continue
		}

		// Overlay is not allowed to have targets that are directories.
		// And we hide symlinks, although it's not clear it helps callers.
		cinfo, err := os.Stat(cinfo.actual)
		if err != nil {
			all[cname] = fs.FileInfoToDirEntry(missingFile(cname))
			continue
		}
		if cinfo.IsDir() {
			return nil, &fs.PathError{Op: "read", Path: name, Err: fmt.Errorf("overlay maps child %s to directory", cname)}
		}
		all[cname] = fs.FileInfoToDirEntry(fakeFile{cname, cinfo})
	}

	// Rebuild list using same storage.
	dirs = dirs[:0]
	for _, d := range all {
		dirs = append(dirs, d)
	}
	slices.SortFunc(dirs, func(x, y fs.DirEntry) int { return strings.Compare(x.Name(), y.Name()) })
	return dirs, nil
}

// Actual returns the actual file system path for the named file.
// It returns the empty string if name has been deleted in the virtual file system.
func Actual(name string) string {
	info := stat(name)
	if info.deleted {
		return ""
	}
	if info.dir || info.replaced {
		return info.actual
	}
	return name
}

// Replaced reports whether the named file has been modified
// in the virtual file system compared to the OS file system.
func Replaced(name string) bool {
	p, ok := overlay[abs(name)]
	return ok && !p.isDir()
}

// Open opens the named file in the virtual file system.
// It must be an ordinary file, not a directory.
func Open(name string) (*os.File, error) {
	Trace("Open", name)

	bad := func(msg string) (*os.File, error) {
		return nil, &fs.PathError{
			Op:   "Open",
			Path: name,
			Err:  errors.New(msg),
		}
	}

	info := stat(name)
	if info.deleted {
		return bad("deleted in overlay")
	}
	if info.dir {
		return bad("cannot open directory in overlay")
	}
	if info.replaced {
		name = info.actual
	}

	return os.Open(name)
}

// ReadFile reads the named file from the virtual file system
// and returns the contents.
func ReadFile(name string) ([]byte, error) {
	f, err := Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return io.ReadAll(f)
}

// IsGoDir reports whether the named directory in the virtual file system
// is a directory containing one or more Go source files.
func IsGoDir(name string) (bool, error) {
	Trace("IsGoDir", name)
	fis, err := ReadDir(name)
	if os.IsNotExist(err) || errors.Is(err, errNotDir) {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	var firstErr error
	for _, d := range fis {
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".go") {
			continue
		}
		if d.Type().IsRegular() {
			return true, nil
		}

		// d is a non-directory, non-regular .go file.
		// Stat to see if it is a symlink, which we allow.
		if actual := Actual(filepath.Join(name, d.Name())); actual != "" {
			fi, err := os.Stat(actual)
			if err == nil && fi.Mode().IsRegular() {
				return true, nil
			}
			if err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}

	// No go files found in directory.
	return false, firstErr
}

// Lstat returns a FileInfo describing the named file in the virtual file system.
// It does not follow symbolic links
func Lstat(name string) (fs.FileInfo, error) {
	Trace("Lstat", name)
	return overlayStat("lstat", name, os.Lstat)
}

// Stat returns a FileInfo describing the named file in the virtual file system.
// It follows symbolic links.
func Stat(name string) (fs.FileInfo, error) {
	Trace("Stat", name)
	return overlayStat("stat", name, os.Stat)
}

// overlayStat implements lstat or Stat (depending on whether os.Lstat or os.Stat is passed in).
func overlayStat(op, path string, osStat func(string) (fs.FileInfo, error)) (fs.FileInfo, error) {
	info := stat(path)
	if info.deleted {
		return nil, &fs.PathError{Op: op, Path: path, Err: fs.ErrNotExist}
	}
	if info.dir {
		return fakeDir(filepath.Base(path)), nil
	}
	if info.replaced {
		// To keep the data model simple, if the overlay contains a symlink we
		// always stat through it (using Stat, not Lstat). That way we don't need to
		// worry about the interaction between Lstat and directories: if a symlink
		// in the overlay points to a directory, we reject it like an ordinary
		// directory.
		ainfo, err := os.Stat(info.actual)
		if err != nil {
			return nil, err
		}
		if ainfo.IsDir() {
			return nil, &fs.PathError{Op: op, Path: path, Err: fmt.Errorf("overlay maps to directory")}
		}
		return fakeFile{name: filepath.Base(path), real: ainfo}, nil
	}
	return osStat(path)
}

// fakeFile provides an fs.FileInfo implementation for an overlaid file,
// so that the file has the name of the overlaid file, but takes all
// other characteristics of the replacement file.
type fakeFile struct {
	name string
	real fs.FileInfo
}

func (f fakeFile) Name() string       { return f.name }
func (f fakeFile) Size() int64        { return f.real.Size() }
func (f fakeFile) Mode() fs.FileMode  { return f.real.Mode() }
func (f fakeFile) ModTime() time.Time { return f.real.ModTime() }
func (f fakeFile) IsDir() bool        { return f.real.IsDir() }
func (f fakeFile) Sys() any           { return f.real.Sys() }

func (f fakeFile) String() string {
	return fs.FormatFileInfo(f)
}

// missingFile provides an fs.FileInfo for an overlaid file where the
// destination file in the overlay doesn't exist. It returns zero values
// for the fileInfo methods other than Name, set to the file's name, and Mode
// set to ModeIrregular.
type missingFile string

func (f missingFile) Name() string       { return string(f) }
func (f missingFile) Size() int64        { return 0 }
func (f missingFile) Mode() fs.FileMode  { return fs.ModeIrregular }
func (f missingFile) ModTime() time.Time { return time.Unix(0, 0) }
func (f missingFile) IsDir() bool        { return false }
func (f missingFile) Sys() any           { return nil }

func (f missingFile) String() string {
	return fs.FormatFileInfo(f)
}

// fakeDir provides an fs.FileInfo implementation for directories that are
// implicitly created by overlaid files. Each directory in the
// path of an overlaid file is considered to exist in the overlay filesystem.
type fakeDir string

func (f fakeDir) Name() string       { return string(f) }
func (f fakeDir) Size() int64        { return 0 }
func (f fakeDir) Mode() fs.FileMode  { return fs.ModeDir | 0500 }
func (f fakeDir) ModTime() time.Time { return time.Unix(0, 0) }
func (f fakeDir) IsDir() bool        { return true }
func (f fakeDir) Sys() any           { return nil }

func (f fakeDir) String() string {
	return fs.FormatFileInfo(f)
}

func cmp(x, y string) int {
	for i := 0; i < len(x) && i < len(y); i++ {
		xi := int(x[i])
		yi := int(y[i])
		if xi == filepath.Separator {
			xi = -1
		}
		if yi == filepath.Separator {
			yi = -1
		}
		if xi != yi {
			return xi - yi
		}
	}
	return len(x) - len(y)
}
