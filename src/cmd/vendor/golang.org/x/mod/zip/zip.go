// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package zip provides functions for creating and extracting module zip files.
//
// Module zip files have several restrictions listed below. These are necessary
// to ensure that module zip files can be extracted consistently on supported
// platforms and file systems.
//
// • All file paths within a zip file must start with "<module>@<version>/",
// where "<module>" is the module path and "<version>" is the version.
// The module path must be valid (see golang.org/x/mod/module.CheckPath).
// The version must be valid and canonical (see
// golang.org/x/mod/module.CanonicalVersion). The path must have a major
// version suffix consistent with the version (see
// golang.org/x/mod/module.Check). The part of the file path after the
// "<module>@<version>/" prefix must be valid (see
// golang.org/x/mod/module.CheckFilePath).
//
// • No two file paths may be equal under Unicode case-folding (see
// strings.EqualFold).
//
// • A go.mod file may or may not appear in the top-level directory. If present,
// it must be named "go.mod", not any other case. Files named "go.mod"
// are not allowed in any other directory.
//
// • The total size in bytes of a module zip file may be at most MaxZipFile
// bytes (500 MiB). The total uncompressed size of the files within the
// zip may also be at most MaxZipFile bytes.
//
// • Each file's uncompressed size must match its declared 64-bit uncompressed
// size in the zip file header.
//
// • If the zip contains files named "<module>@<version>/go.mod" or
// "<module>@<version>/LICENSE", their sizes in bytes may be at most
// MaxGoMod or MaxLICENSE, respectively (both are 16 MiB).
//
// • Empty directories are ignored. File permissions and timestamps are also
// ignored.
//
// • Symbolic links and other irregular files are not allowed.
//
// Note that this package does not provide hashing functionality. See
// golang.org/x/mod/sumdb/dirhash.
package zip

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/mod/module"
)

const (
	// MaxZipFile is the maximum size in bytes of a module zip file. The
	// go command will report an error if either the zip file or its extracted
	// content is larger than this.
	MaxZipFile = 500 << 20

	// MaxGoMod is the maximum size in bytes of a go.mod file within a
	// module zip file.
	MaxGoMod = 16 << 20

	// MaxLICENSE is the maximum size in bytes of a LICENSE file within a
	// module zip file.
	MaxLICENSE = 16 << 20
)

// File provides an abstraction for a file in a directory, zip, or anything
// else that looks like a file.
type File interface {
	// Path returns a clean slash-separated relative path from the module root
	// directory to the file.
	Path() string

	// Lstat returns information about the file. If the file is a symbolic link,
	// Lstat returns information about the link itself, not the file it points to.
	Lstat() (os.FileInfo, error)

	// Open provides access to the data within a regular file. Open may return
	// an error if called on a directory or symbolic link.
	Open() (io.ReadCloser, error)
}

// Create builds a zip archive for module m from an abstract list of files
// and writes it to w.
//
// Create verifies the restrictions described in the package documentation
// and should not produce an archive that Unzip cannot extract. Create does not
// include files in the output archive if they don't belong in the module zip.
// In particular, Create will not include files in modules found in
// subdirectories, most files in vendor directories, or irregular files (such
// as symbolic links) in the output archive.
func Create(w io.Writer, m module.Version, files []File) (err error) {
	defer func() {
		if err != nil {
			err = &zipError{verb: "create zip", err: err}
		}
	}()

	// Check that the version is canonical, the module path is well-formed, and
	// the major version suffix matches the major version.
	if vers := module.CanonicalVersion(m.Version); vers != m.Version {
		return fmt.Errorf("version %q is not canonical (should be %q)", m.Version, vers)
	}
	if err := module.Check(m.Path, m.Version); err != nil {
		return err
	}

	// Find directories containing go.mod files (other than the root).
	// These directories will not be included in the output zip.
	haveGoMod := make(map[string]bool)
	for _, f := range files {
		dir, base := path.Split(f.Path())
		if strings.EqualFold(base, "go.mod") {
			info, err := f.Lstat()
			if err != nil {
				return err
			}
			if info.Mode().IsRegular() {
				haveGoMod[dir] = true
			}
		}
	}

	inSubmodule := func(p string) bool {
		for {
			dir, _ := path.Split(p)
			if dir == "" {
				return false
			}
			if haveGoMod[dir] {
				return true
			}
			p = dir[:len(dir)-1]
		}
	}

	// Create the module zip file.
	zw := zip.NewWriter(w)
	prefix := fmt.Sprintf("%s@%s/", m.Path, m.Version)

	addFile := func(f File, path string, size int64) error {
		rc, err := f.Open()
		if err != nil {
			return err
		}
		defer rc.Close()
		w, err := zw.Create(prefix + path)
		if err != nil {
			return err
		}
		lr := &io.LimitedReader{R: rc, N: size + 1}
		if _, err := io.Copy(w, lr); err != nil {
			return err
		}
		if lr.N <= 0 {
			return fmt.Errorf("file %q is larger than declared size", path)
		}
		return nil
	}

	collisions := make(collisionChecker)
	maxSize := int64(MaxZipFile)
	for _, f := range files {
		p := f.Path()
		if p != path.Clean(p) {
			return fmt.Errorf("file path %s is not clean", p)
		}
		if path.IsAbs(p) {
			return fmt.Errorf("file path %s is not relative", p)
		}
		if isVendoredPackage(p) || inSubmodule(p) {
			continue
		}
		if p == ".hg_archival.txt" {
			// Inserted by hg archive.
			// The go command drops this regardless of the VCS being used.
			continue
		}
		if err := module.CheckFilePath(p); err != nil {
			return err
		}
		if strings.ToLower(p) == "go.mod" && p != "go.mod" {
			return fmt.Errorf("found file named %s, want all lower-case go.mod", p)
		}
		info, err := f.Lstat()
		if err != nil {
			return err
		}
		if err := collisions.check(p, info.IsDir()); err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			// Skip symbolic links (golang.org/issue/27093).
			continue
		}
		size := info.Size()
		if size < 0 || maxSize < size {
			return fmt.Errorf("module source tree too large (max size is %d bytes)", MaxZipFile)
		}
		maxSize -= size
		if p == "go.mod" && size > MaxGoMod {
			return fmt.Errorf("go.mod file too large (max size is %d bytes)", MaxGoMod)
		}
		if p == "LICENSE" && size > MaxLICENSE {
			return fmt.Errorf("LICENSE file too large (max size is %d bytes)", MaxLICENSE)
		}

		if err := addFile(f, p, size); err != nil {
			return err
		}
	}

	return zw.Close()
}

// CreateFromDir creates a module zip file for module m from the contents of
// a directory, dir. The zip content is written to w.
//
// CreateFromDir verifies the restrictions described in the package
// documentation and should not produce an archive that Unzip cannot extract.
// CreateFromDir does not include files in the output archive if they don't
// belong in the module zip. In particular, CreateFromDir will not include
// files in modules found in subdirectories, most files in vendor directories,
// or irregular files (such as symbolic links) in the output archive.
func CreateFromDir(w io.Writer, m module.Version, dir string) (err error) {
	defer func() {
		if zerr, ok := err.(*zipError); ok {
			zerr.path = dir
		} else if err != nil {
			err = &zipError{verb: "create zip", path: dir, err: err}
		}
	}()

	var files []File
	err = filepath.Walk(dir, func(filePath string, info os.FileInfo, err error) error {
		relPath, err := filepath.Rel(dir, filePath)
		if err != nil {
			return err
		}
		slashPath := filepath.ToSlash(relPath)

		if info.IsDir() {
			if filePath == dir {
				// Don't skip the top-level directory.
				return nil
			}

			// Skip some subdirectories inside vendor, but maintain bug
			// golang.org/issue/31562, described in isVendoredPackage.
			// We would like Create and CreateFromDir to produce the same result
			// for a set of files, whether expressed as a directory tree or zip.
			if isVendoredPackage(slashPath) {
				return filepath.SkipDir
			}

			// Skip submodules (directories containing go.mod files).
			if goModInfo, err := os.Lstat(filepath.Join(filePath, "go.mod")); err == nil && !goModInfo.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if info.Mode().IsRegular() {
			if !isVendoredPackage(slashPath) {
				files = append(files, dirFile{
					filePath:  filePath,
					slashPath: slashPath,
					info:      info,
				})
			}
			return nil
		}

		// Not a regular file or a directory. Probably a symbolic link.
		// Irregular files are ignored, so skip it.
		return nil
	})
	if err != nil {
		return err
	}

	return Create(w, m, files)
}

type dirFile struct {
	filePath, slashPath string
	info                os.FileInfo
}

func (f dirFile) Path() string                 { return f.slashPath }
func (f dirFile) Lstat() (os.FileInfo, error)  { return f.info, nil }
func (f dirFile) Open() (io.ReadCloser, error) { return os.Open(f.filePath) }

func isVendoredPackage(name string) bool {
	var i int
	if strings.HasPrefix(name, "vendor/") {
		i += len("vendor/")
	} else if j := strings.Index(name, "/vendor/"); j >= 0 {
		// This offset looks incorrect; this should probably be
		//
		// 	i = j + len("/vendor/")
		//
		// (See https://golang.org/issue/31562.)
		//
		// Unfortunately, we can't fix it without invalidating checksums.
		// Fortunately, the error appears to be strictly conservative: we'll retain
		// vendored packages that we should have pruned, but we won't prune
		// non-vendored packages that we should have retained.
		//
		// Since this defect doesn't seem to break anything, it's not worth fixing
		// for now.
		i += len("/vendor/")
	} else {
		return false
	}
	return strings.Contains(name[i:], "/")
}

// Unzip extracts the contents of a module zip file to a directory.
//
// Unzip checks all restrictions listed in the package documentation and returns
// an error if the zip archive is not valid. In some cases, files may be written
// to dir before an error is returned (for example, if a file's uncompressed
// size does not match its declared size).
//
// dir may or may not exist: Unzip will create it and any missing parent
// directories if it doesn't exist. If dir exists, it must be empty.
func Unzip(dir string, m module.Version, zipFile string) (err error) {
	defer func() {
		if err != nil {
			err = &zipError{verb: "unzip", path: zipFile, err: err}
		}
	}()

	if vers := module.CanonicalVersion(m.Version); vers != m.Version {
		return fmt.Errorf("version %q is not canonical (should be %q)", m.Version, vers)
	}
	if err := module.Check(m.Path, m.Version); err != nil {
		return err
	}

	// Check that the directory is empty. Don't create it yet in case there's
	// an error reading the zip.
	files, _ := ioutil.ReadDir(dir)
	if len(files) > 0 {
		return fmt.Errorf("target directory %v exists and is not empty", dir)
	}

	// Open the zip file and ensure it's under the size limit.
	f, err := os.Open(zipFile)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	zipSize := info.Size()
	if zipSize > MaxZipFile {
		return fmt.Errorf("module zip file is too large (%d bytes; limit is %d bytes)", zipSize, MaxZipFile)
	}

	z, err := zip.NewReader(f, zipSize)
	if err != nil {
		return err
	}

	// Check total size, valid file names.
	collisions := make(collisionChecker)
	prefix := fmt.Sprintf("%s@%s/", m.Path, m.Version)
	var size int64
	for _, zf := range z.File {
		if !strings.HasPrefix(zf.Name, prefix) {
			return fmt.Errorf("unexpected file name %s", zf.Name)
		}
		name := zf.Name[len(prefix):]
		if name == "" {
			continue
		}
		isDir := strings.HasSuffix(name, "/")
		if isDir {
			name = name[:len(name)-1]
		}
		if path.Clean(name) != name {
			return fmt.Errorf("invalid file name %s", zf.Name)
		}
		if err := module.CheckFilePath(name); err != nil {
			return err
		}
		if err := collisions.check(name, isDir); err != nil {
			return err
		}
		if isDir {
			continue
		}
		if base := path.Base(name); strings.EqualFold(base, "go.mod") {
			if base != name {
				return fmt.Errorf("found go.mod file not in module root directory (%s)", zf.Name)
			} else if name != "go.mod" {
				return fmt.Errorf("found file named %s, want all lower-case go.mod", zf.Name)
			}
		}
		s := int64(zf.UncompressedSize64)
		if s < 0 || MaxZipFile-size < s {
			return fmt.Errorf("total uncompressed size of module contents too large (max size is %d bytes)", MaxZipFile)
		}
		size += s
		if name == "go.mod" && s > MaxGoMod {
			return fmt.Errorf("go.mod file too large (max size is %d bytes)", MaxGoMod)
		}
		if name == "LICENSE" && s > MaxLICENSE {
			return fmt.Errorf("LICENSE file too large (max size is %d bytes)", MaxLICENSE)
		}
	}

	// Unzip, enforcing sizes checked earlier.
	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}
	for _, zf := range z.File {
		name := zf.Name[len(prefix):]
		if name == "" || strings.HasSuffix(name, "/") {
			continue
		}
		dst := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(dst), 0777); err != nil {
			return err
		}
		w, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0444)
		if err != nil {
			return err
		}
		r, err := zf.Open()
		if err != nil {
			w.Close()
			return err
		}
		lr := &io.LimitedReader{R: r, N: int64(zf.UncompressedSize64) + 1}
		_, err = io.Copy(w, lr)
		r.Close()
		if err != nil {
			w.Close()
			return err
		}
		if err := w.Close(); err != nil {
			return err
		}
		if lr.N <= 0 {
			return fmt.Errorf("uncompressed size of file %s is larger than declared size (%d bytes)", zf.Name, zf.UncompressedSize64)
		}
	}

	return nil
}

// collisionChecker finds case-insensitive name collisions and paths that
// are listed as both files and directories.
//
// The keys of this map are processed with strToFold. pathInfo has the original
// path for each folded path.
type collisionChecker map[string]pathInfo

type pathInfo struct {
	path  string
	isDir bool
}

func (cc collisionChecker) check(p string, isDir bool) error {
	fold := strToFold(p)
	if other, ok := cc[fold]; ok {
		if p != other.path {
			return fmt.Errorf("case-insensitive file name collision: %q and %q", other.path, p)
		}
		if isDir != other.isDir {
			return fmt.Errorf("entry %q is both a file and a directory", p)
		}
		if !isDir {
			return fmt.Errorf("multiple entries for file %q", p)
		}
		// It's not an error if check is called with the same directory multiple
		// times. check is called recursively on parent directories, so check
		// may be called on the same directory many times.
	} else {
		cc[fold] = pathInfo{path: p, isDir: isDir}
	}

	if parent := path.Dir(p); parent != "." {
		return cc.check(parent, true)
	}
	return nil
}

type zipError struct {
	verb, path string
	err        error
}

func (e *zipError) Error() string {
	if e.path == "" {
		return fmt.Sprintf("%s: %v", e.verb, e.err)
	} else {
		return fmt.Sprintf("%s %s: %v", e.verb, e.path, e.err)
	}
}

func (e *zipError) Unwrap() error {
	return e.err
}

// strToFold returns a string with the property that
//	strings.EqualFold(s, t) iff strToFold(s) == strToFold(t)
// This lets us test a large set of strings for fold-equivalent
// duplicates without making a quadratic number of calls
// to EqualFold. Note that strings.ToUpper and strings.ToLower
// do not have the desired property in some corner cases.
func strToFold(s string) string {
	// Fast path: all ASCII, no upper case.
	// Most paths look like this already.
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= utf8.RuneSelf || 'A' <= c && c <= 'Z' {
			goto Slow
		}
	}
	return s

Slow:
	var buf bytes.Buffer
	for _, r := range s {
		// SimpleFold(x) cycles to the next equivalent rune > x
		// or wraps around to smaller values. Iterate until it wraps,
		// and we've found the minimum value.
		for {
			r0 := r
			r = unicode.SimpleFold(r0)
			if r <= r0 {
				break
			}
		}
		// Exception to allow fast path above: A-Z => a-z
		if 'A' <= r && r <= 'Z' {
			r += 'a' - 'A'
		}
		buf.WriteRune(r)
	}
	return buf.String()
}
