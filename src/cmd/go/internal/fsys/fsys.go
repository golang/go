// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fsys is an abstraction for reading files that
// allows for virtual overlays on top of the files on disk.
package fsys

import (
	"encoding/json"
	"errors"
	"fmt"
	"internal/godebug"
	"io"
	"io/fs"
	"log"
	"os"
	pathpkg "path"
	"path/filepath"
	"runtime/debug"
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

// OverlayFile is the path to a text file in the OverlayJSON format.
// It is the value of the -overlay flag.
var OverlayFile string

// OverlayJSON is the format overlay files are expected to be in.
// The Replace map maps from overlaid paths to replacement paths:
// the Go command will forward all reads trying to open
// each overlaid path to its replacement path, or consider the overlaid
// path not to exist if the replacement path is empty.
type OverlayJSON struct {
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
var cwd string               // copy of base.Cwd() to avoid dependency

// canonicalize a path for looking it up in the overlay.
// Important: filepath.Join(cwd, path) doesn't always produce
// the correct absolute path if path is relative, because on
// Windows producing the correct absolute path requires making
// a syscall. So this should only be used when looking up paths
// in the overlay, or canonicalizing the paths in the overlay.
func canonicalize(path string) string {
	if path == "" {
		return ""
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}

	if v := filepath.VolumeName(cwd); v != "" && path[0] == filepath.Separator {
		// On Windows filepath.Join(cwd, path) doesn't always work. In general
		// filepath.Abs needs to make a syscall on Windows. Elsewhere in cmd/go
		// use filepath.Join(cwd, path), but cmd/go specifically supports Windows
		// paths that start with "\" which implies the path is relative to the
		// volume of the working directory. See golang.org/issue/8130.
		return filepath.Join(v, path)
	}

	// Make the path absolute.
	return filepath.Join(cwd, path)
}

// Init initializes the overlay, if one is being used.
func Init(wd string) error {
	if overlay != nil {
		// already initialized
		return nil
	}

	cwd = wd

	if OverlayFile == "" {
		return nil
	}

	Trace("ReadFile", OverlayFile)
	b, err := os.ReadFile(OverlayFile)
	if err != nil {
		return fmt.Errorf("reading overlay file: %v", err)
	}

	var overlayJSON OverlayJSON
	if err := json.Unmarshal(b, &overlayJSON); err != nil {
		return fmt.Errorf("parsing overlay JSON: %v", err)
	}

	return initFromJSON(overlayJSON)
}

func initFromJSON(overlayJSON OverlayJSON) error {
	// Canonicalize the paths in the overlay map.
	// Use reverseCanonicalized to check for collisions:
	// no two 'from' paths should canonicalize to the same path.
	overlay = make(map[string]*node)
	reverseCanonicalized := make(map[string]string) // inverse of canonicalize operation, to check for duplicates
	// Build a table of file and directory nodes from the replacement map.

	// Remove any potential non-determinism from iterating over map by sorting it.
	replaceFrom := make([]string, 0, len(overlayJSON.Replace))
	for k := range overlayJSON.Replace {
		replaceFrom = append(replaceFrom, k)
	}
	sort.Strings(replaceFrom)

	for _, from := range replaceFrom {
		to := overlayJSON.Replace[from]
		// Canonicalize paths and check for a collision.
		if from == "" {
			return fmt.Errorf("empty string key in overlay file Replace map")
		}
		cfrom := canonicalize(from)
		if to != "" {
			// Don't canonicalize "", meaning to delete a file, because then it will turn into ".".
			to = canonicalize(to)
		}
		if otherFrom, seen := reverseCanonicalized[cfrom]; seen {
			return fmt.Errorf(
				"paths %q and %q both canonicalize to %q in overlay file Replace map", otherFrom, from, cfrom)
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
	path = canonicalize(path)

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

// errNotDir is used to communicate from ReadDir to IsDirWithGoFiles
// that the argument is not a directory, so that IsDirWithGoFiles doesn't
// return an error.
var errNotDir = errors.New("not a directory")

func nonFileInOverlayError(overlayPath string) error {
	return fmt.Errorf("replacement path %q is a directory, not a file", overlayPath)
}

// readDir reads a dir on disk, returning an error that is errNotDir if the dir is not a directory.
// Unfortunately, the error returned by os.ReadDir if dir is not a directory
// can vary depending on the OS (Linux, Mac, Windows return ENOTDIR; BSD returns EINVAL).
func readDir(dir string) ([]fs.FileInfo, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, err
		}
		if dirfi, staterr := os.Stat(dir); staterr == nil && !dirfi.IsDir() {
			return nil, &fs.PathError{Op: "ReadDir", Path: dir, Err: errNotDir}
		}
		return nil, err
	}

	fis := make([]fs.FileInfo, 0, len(entries))
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}
		fis = append(fis, info)
	}
	return fis, nil
}

// ReadDir provides a slice of fs.FileInfo entries corresponding
// to the overlaid files in the directory.
func ReadDir(dir string) ([]fs.FileInfo, error) {
	Trace("ReadDir", dir)
	dir = canonicalize(dir)
	if _, ok := parentIsOverlayFile(dir); ok {
		return nil, &fs.PathError{Op: "ReadDir", Path: dir, Err: errNotDir}
	}

	dirNode := overlay[dir]
	if dirNode == nil {
		return readDir(dir)
	}
	if dirNode.isDeleted() {
		return nil, &fs.PathError{Op: "ReadDir", Path: dir, Err: fs.ErrNotExist}
	}
	diskfis, err := readDir(dir)
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

// OverlayPath returns the path to the overlaid contents of the
// file, the empty string if the overlay deletes the file, or path
// itself if the file is not in the overlay, the file is a directory
// in the overlay, or there is no overlay.
// It returns true if the path is overlaid with a regular file
// or deleted, and false otherwise.
func OverlayPath(path string) (string, bool) {
	if p, ok := overlay[canonicalize(path)]; ok && !p.isDir() {
		return p.actualFilePath, ok
	}

	return path, false
}

// Open opens the file at or overlaid on the given path.
func Open(path string) (*os.File, error) {
	Trace("Open", path)
	return openFile(path, os.O_RDONLY, 0)
}

func openFile(path string, flag int, perm os.FileMode) (*os.File, error) {
	cpath := canonicalize(path)
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

// ReadFile reads the file at or overlaid on the given path.
func ReadFile(path string) ([]byte, error) {
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return io.ReadAll(f)
}

// IsDirWithGoFiles reports whether dir is a directory containing Go files
// either on disk or in the overlay.
func IsDirWithGoFiles(dir string) (bool, error) {
	Trace("IsDirWithGoFiles", dir)
	fis, err := ReadDir(dir)
	if os.IsNotExist(err) || errors.Is(err, errNotDir) {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	var firstErr error
	for _, fi := range fis {
		if fi.IsDir() {
			continue
		}

		// TODO(matloob): this enforces that the "from" in the map
		// has a .go suffix, but the actual destination file
		// doesn't need to have a .go suffix. Is this okay with the
		// compiler?
		if !strings.HasSuffix(fi.Name(), ".go") {
			continue
		}
		if fi.Mode().IsRegular() {
			return true, nil
		}

		// fi is the result of an Lstat, so it doesn't follow symlinks.
		// But it's okay if the file is a symlink pointing to a regular
		// file, so use os.Stat to follow symlinks and check that.
		actualFilePath, _ := OverlayPath(filepath.Join(dir, fi.Name()))
		fi, err := os.Stat(actualFilePath)
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

// Lstat implements a version of os.Lstat that operates on the overlay filesystem.
func Lstat(path string) (fs.FileInfo, error) {
	Trace("Lstat", path)
	return overlayStat(path, os.Lstat, "lstat")
}

// Stat implements a version of os.Stat that operates on the overlay filesystem.
func Stat(path string) (fs.FileInfo, error) {
	Trace("Stat", path)
	return overlayStat(path, os.Stat, "stat")
}

// overlayStat implements lstat or Stat (depending on whether os.Lstat or os.Stat is passed in).
func overlayStat(path string, osStat func(string) (fs.FileInfo, error), opName string) (fs.FileInfo, error) {
	cpath := canonicalize(path)

	if _, ok := parentIsOverlayFile(filepath.Dir(cpath)); ok {
		return nil, &fs.PathError{Op: opName, Path: cpath, Err: fs.ErrNotExist}
	}

	node, ok := overlay[cpath]
	if !ok {
		// The file or directory is not overlaid.
		return osStat(path)
	}

	switch {
	case node.isDeleted():
		return nil, &fs.PathError{Op: opName, Path: cpath, Err: fs.ErrNotExist}
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
			return nil, &fs.PathError{Op: opName, Path: cpath, Err: nonFileInOverlayError(node.actualFilePath)}
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
