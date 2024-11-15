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
	"log"
	"maps"
	"os"
	pathpkg "path"
	"path/filepath"
	"runtime/debug"
	"slices"
	"sort"
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

type node struct {
	actualFilePath string           // empty if a directory
	children       map[string]*node // path element → file or directory
}

func (n *node) isDir() bool {
	return n.actualFilePath == "" && n.children != nil
}

func (n *node) isDeleted() bool {
	return n.actualFilePath == "" && n.children == nil
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
		return err
	}

	// Canonicalize the paths in the overlay map.
	// Use reverseCanonicalized to check for collisions:
	// no two 'from' paths should abs to the same path.
	overlay = make(map[string]*node)
	reverseCanonicalized := make(map[string]string) // inverse of abs operation, to check for duplicates
	// Build a table of file and directory nodes from the replacement map.

	for _, from := range slices.Sorted(maps.Keys(ojs.Replace)) {
		to := ojs.Replace[from]
		// Canonicalize paths and check for a collision.
		if from == "" {
			return fmt.Errorf("empty string key in overlay file Replace map")
		}
		cfrom := abs(from)
		if to != "" {
			// Don't abs "", meaning to delete a file, because then it will turn into ".".
			to = abs(to)
		}
		if otherFrom, seen := reverseCanonicalized[cfrom]; seen {
			return fmt.Errorf(
				"paths %q and %q both abs to %q in overlay file Replace map", otherFrom, from, cfrom)
		}
		reverseCanonicalized[cfrom] = from
		from = cfrom

		// Create node for overlaid file.
		dir, base := filepath.Dir(from), filepath.Base(from)
		if n, ok := overlay[from]; ok {
			// All 'from' paths in the overlay are file paths. Since the from paths
			// are in a map, they are unique, so if the node already exists we added
			// it below when we create parent directory nodes. That is, that
			// both a file and a path to one of its parent directories exist as keys
			// in the Replace map.
			//
			// This only applies if the overlay directory has any files or directories
			// in it: placeholder directories that only contain deleted files don't
			// count. They are safe to be overwritten with actual files.
			for _, f := range n.children {
				if !f.isDeleted() {
					return fmt.Errorf("invalid overlay: path %v is used as both file and directory", from)
				}
			}
		}
		overlay[from] = &node{actualFilePath: to}

		// Add parent directory nodes to overlay structure.
		childNode := overlay[from]
		for {
			dirNode := overlay[dir]
			if dirNode == nil || dirNode.isDeleted() {
				dirNode = &node{children: make(map[string]*node)}
				overlay[dir] = dirNode
			}
			if childNode.isDeleted() {
				// Only create one parent for a deleted file:
				// the directory only conditionally exists if
				// there are any non-deleted children, so
				// we don't create their parents.
				if dirNode.isDir() {
					dirNode.children[base] = childNode
				}
				break
			}
			if !dirNode.isDir() {
				// This path already exists as a file, so it can't be a parent
				// directory. See comment at error above.
				return fmt.Errorf("invalid overlay: path %v is used as both file and directory", dir)
			}
			dirNode.children[base] = childNode
			parent := filepath.Dir(dir)
			if parent == dir {
				break // reached the top; there is no parent
			}
			dir, base = parent, filepath.Base(dir)
			childNode = dirNode
		}
	}

	return nil
}

// IsDir returns true if path is a directory on disk or in the
// overlay.
func IsDir(path string) (bool, error) {
	Trace("IsDir", path)
	path = abs(path)

	if _, ok := parentIsOverlayFile(path); ok {
		return false, nil
	}

	if n, ok := overlay[path]; ok {
		return n.isDir(), nil
	}

	fi, err := os.Stat(path)
	if err != nil {
		return false, err
	}

	return fi.IsDir(), nil
}

// parentIsOverlayFile returns whether name or any of
// its parents are files in the overlay, and the first parent found,
// including name itself, that's a file in the overlay.
func parentIsOverlayFile(name string) (string, bool) {
	if overlay != nil {
		// Check if name can't possibly be a directory because
		// it or one of its parents is overlaid with a file.
		// TODO(matloob): Maybe save this to avoid doing it every time?
		prefix := name
		for {
			node := overlay[prefix]
			if node != nil && !node.isDir() {
				return prefix, true
			}
			parent := filepath.Dir(prefix)
			if parent == prefix {
				break
			}
			prefix = parent
		}
	}

	return "", false
}

// errNotDir is used to communicate from ReadDir to IsGoDir
// that the argument is not a directory, so that IsGoDir doesn't
// return an error.
var errNotDir = errors.New("not a directory")

func nonFileInOverlayError(overlayPath string) error {
	return fmt.Errorf("replacement path %q is a directory, not a file", overlayPath)
}

// osReadDir is like os.ReadDir but returns []fs.FileInfo and corrects the error to be errNotDir
// if the problem is that name exists but is not a directory.
func osReadDir(name string) ([]fs.FileInfo, error) {
	dirs, err := os.ReadDir(name)
	if err != nil && !os.IsNotExist(err) {
		if info, err := os.Stat(name); err == nil && !info.IsDir() {
			return nil, &fs.PathError{Op: "ReadDir", Path: name, Err: errNotDir}
		}
	}

	// Convert dirs to infos, even if there is an error,
	// so that we preserve any partial read from os.ReadDir.
	infos := make([]fs.FileInfo, 0, len(dirs))
	for _, dir := range dirs {
		info, err := dir.Info()
		if err != nil {
			continue
		}
		infos = append(infos, info)
	}

	return infos, err
}

// ReadDir reads the named directory in the virtual file system.
func ReadDir(dir string) ([]fs.FileInfo, error) {
	Trace("ReadDir", dir)
	dir = abs(dir)
	if _, ok := parentIsOverlayFile(dir); ok {
		return nil, &fs.PathError{Op: "ReadDir", Path: dir, Err: errNotDir}
	}

	dirNode := overlay[dir]
	if dirNode == nil {
		return osReadDir(dir)
	}
	if dirNode.isDeleted() {
		return nil, &fs.PathError{Op: "ReadDir", Path: dir, Err: fs.ErrNotExist}
	}
	diskfis, err := osReadDir(dir)
	if err != nil && !os.IsNotExist(err) && !errors.Is(err, errNotDir) {
		return nil, err
	}

	// Stat files in overlay to make composite list of fileinfos
	files := make(map[string]fs.FileInfo)
	for _, f := range diskfis {
		files[f.Name()] = f
	}
	for name, to := range dirNode.children {
		switch {
		case to.isDir():
			files[name] = fakeDir(name)
		case to.isDeleted():
			delete(files, name)
		default:
			// To keep the data model simple, if the overlay contains a symlink we
			// always stat through it (using Stat, not Lstat). That way we don't need
			// to worry about the interaction between Lstat and directories: if a
			// symlink in the overlay points to a directory, we reject it like an
			// ordinary directory.
			fi, err := os.Stat(to.actualFilePath)
			if err != nil {
				files[name] = missingFile(name)
				continue
			} else if fi.IsDir() {
				return nil, &fs.PathError{Op: "Stat", Path: filepath.Join(dir, name), Err: nonFileInOverlayError(to.actualFilePath)}
			}
			// Add a fileinfo for the overlaid file, so that it has
			// the original file's name, but the overlaid file's metadata.
			files[name] = fakeFile{name, fi}
		}
	}
	sortedFiles := diskfis[:0]
	for _, f := range files {
		sortedFiles = append(sortedFiles, f)
	}
	sort.Slice(sortedFiles, func(i, j int) bool { return sortedFiles[i].Name() < sortedFiles[j].Name() })
	return sortedFiles, nil
}

// Actual returns the actual file system path for the named file.
// It returns the empty string if name has been deleted in the virtual file system.
func Actual(name string) string {
	if p, ok := overlay[abs(name)]; ok && !p.isDir() {
		return p.actualFilePath
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
	return openFile(name, os.O_RDONLY, 0)
}

func openFile(path string, flag int, perm os.FileMode) (*os.File, error) {
	cpath := abs(path)
	if node, ok := overlay[cpath]; ok {
		// Opening a file in the overlay.
		if node.isDir() {
			return nil, &fs.PathError{Op: "OpenFile", Path: path, Err: errors.New("fsys.OpenFile doesn't support opening directories yet")}
		}
		// We can't open overlaid paths for write.
		if perm != os.FileMode(os.O_RDONLY) {
			return nil, &fs.PathError{Op: "OpenFile", Path: path, Err: errors.New("overlaid files can't be opened for write")}
		}
		return os.OpenFile(node.actualFilePath, flag, perm)
	}
	if parent, ok := parentIsOverlayFile(filepath.Dir(cpath)); ok {
		// The file is deleted explicitly in the Replace map,
		// or implicitly because one of its parent directories was
		// replaced by a file.
		return nil, &fs.PathError{
			Op:   "Open",
			Path: path,
			Err:  fmt.Errorf("file %s does not exist: parent directory %s is replaced by a file in overlay", path, parent),
		}
	}
	return os.OpenFile(cpath, flag, perm)
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
	for _, fi := range fis {
		if fi.IsDir() || !strings.HasSuffix(fi.Name(), ".go") {
			continue
		}
		if fi.Mode().IsRegular() {
			return true, nil
		}

		// fi is the result of an Lstat, so it doesn't follow symlinks.
		// But it's okay if the file is a symlink pointing to a regular
		// file, so use os.Stat to follow symlinks and check that.
		fi, err := os.Stat(Actual(filepath.Join(name, fi.Name())))
		if err == nil && fi.Mode().IsRegular() {
			return true, nil
		}
		if err != nil && firstErr == nil {
			firstErr = err
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
	cpath := abs(path)

	if _, ok := parentIsOverlayFile(filepath.Dir(cpath)); ok {
		return nil, &fs.PathError{Op: op, Path: path, Err: fs.ErrNotExist}
	}

	node, ok := overlay[cpath]
	if !ok {
		// The file or directory is not overlaid.
		return osStat(path)
	}

	switch {
	case node.isDeleted():
		return nil, &fs.PathError{Op: op, Path: path, Err: fs.ErrNotExist}
	case node.isDir():
		return fakeDir(filepath.Base(path)), nil
	default:
		// To keep the data model simple, if the overlay contains a symlink we
		// always stat through it (using Stat, not Lstat). That way we don't need to
		// worry about the interaction between Lstat and directories: if a symlink
		// in the overlay points to a directory, we reject it like an ordinary
		// directory.
		fi, err := os.Stat(node.actualFilePath)
		if err != nil {
			return nil, err
		}
		if fi.IsDir() {
			return nil, &fs.PathError{Op: op, Path: path, Err: nonFileInOverlayError(node.actualFilePath)}
		}
		return fakeFile{name: filepath.Base(path), real: fi}, nil
	}
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
