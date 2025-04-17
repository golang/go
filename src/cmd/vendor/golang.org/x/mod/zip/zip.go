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
// The module path must be valid (see [golang.org/x/mod/module.CheckPath]).
// The version must be valid and canonical (see
// [golang.org/x/mod/module.CanonicalVersion]). The path must have a major
// version suffix consistent with the version (see
// [golang.org/x/mod/module.Check]). The part of the file path after the
// "<module>@<version>/" prefix must be valid (see
// [golang.org/x/mod/module.CheckFilePath]).
//
// • No two file paths may be equal under Unicode case-folding (see
// [strings.EqualFold]).
//
// • A go.mod file may or may not appear in the top-level directory. If present,
// it must be named "go.mod", not any other case. Files named "go.mod"
// are not allowed in any other directory.
//
// • The total size in bytes of a module zip file may be at most [MaxZipFile]
// bytes (500 MiB). The total uncompressed size of the files within the
// zip may also be at most [MaxZipFile] bytes.
//
// • Each file's uncompressed size must match its declared 64-bit uncompressed
// size in the zip file header.
//
// • If the zip contains files named "<module>@<version>/go.mod" or
// "<module>@<version>/LICENSE", their sizes in bytes may be at most
// [MaxGoMod] or [MaxLICENSE], respectively (both are 16 MiB).
//
// • Empty directories are ignored. File permissions and timestamps are also
// ignored.
//
// • Symbolic links and other irregular files are not allowed.
//
// Note that this package does not provide hashing functionality. See
// [golang.org/x/mod/sumdb/dirhash].
package zip

import (
	"archive/zip"
	"bytes"
	"errors"
	"fmt"
	"go/version"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"golang.org/x/mod/modfile"
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

// CheckedFiles reports whether a set of files satisfy the name and size
// constraints required by module zip files. The constraints are listed in the
// package documentation.
//
// Functions that produce this report may include slightly different sets of
// files. See documentation for CheckFiles, CheckDir, and CheckZip for details.
type CheckedFiles struct {
	// Valid is a list of file paths that should be included in a zip file.
	Valid []string

	// Omitted is a list of files that are ignored when creating a module zip
	// file, along with the reason each file is ignored.
	Omitted []FileError

	// Invalid is a list of files that should not be included in a module zip
	// file, along with the reason each file is invalid.
	Invalid []FileError

	// SizeError is non-nil if the total uncompressed size of the valid files
	// exceeds the module zip size limit or if the zip file itself exceeds the
	// limit.
	SizeError error
}

// Err returns an error if [CheckedFiles] does not describe a valid module zip
// file. [CheckedFiles.SizeError] is returned if that field is set.
// A [FileErrorList] is returned
// if there are one or more invalid files. Other errors may be returned in the
// future.
func (cf CheckedFiles) Err() error {
	if cf.SizeError != nil {
		return cf.SizeError
	}
	if len(cf.Invalid) > 0 {
		return FileErrorList(cf.Invalid)
	}
	return nil
}

type FileErrorList []FileError

func (el FileErrorList) Error() string {
	buf := &strings.Builder{}
	sep := ""
	for _, e := range el {
		buf.WriteString(sep)
		buf.WriteString(e.Error())
		sep = "\n"
	}
	return buf.String()
}

type FileError struct {
	Path string
	Err  error
}

func (e FileError) Error() string {
	return fmt.Sprintf("%s: %s", e.Path, e.Err)
}

func (e FileError) Unwrap() error {
	return e.Err
}

var (
	// Predefined error messages for invalid files. Not exhaustive.
	errPathNotClean    = errors.New("file path is not clean")
	errPathNotRelative = errors.New("file path is not relative")
	errGoModCase       = errors.New("go.mod files must have lowercase names")
	errGoModSize       = fmt.Errorf("go.mod file too large (max size is %d bytes)", MaxGoMod)
	errLICENSESize     = fmt.Errorf("LICENSE file too large (max size is %d bytes)", MaxLICENSE)

	// Predefined error messages for omitted files. Not exhaustive.
	errVCS           = errors.New("directory is a version control repository")
	errVendored      = errors.New("file is in vendor directory")
	errSubmoduleFile = errors.New("file is in another module")
	errSubmoduleDir  = errors.New("directory is in another module")
	errHgArchivalTxt = errors.New("file is inserted by 'hg archive' and is always omitted")
	errSymlink       = errors.New("file is a symbolic link")
	errNotRegular    = errors.New("not a regular file")
)

// CheckFiles reports whether a list of files satisfy the name and size
// constraints listed in the package documentation. The returned CheckedFiles
// record contains lists of valid, invalid, and omitted files. Every file in
// the given list will be included in exactly one of those lists.
//
// CheckFiles returns an error if the returned CheckedFiles does not describe
// a valid module zip file (according to CheckedFiles.Err). The returned
// CheckedFiles is still populated when an error is returned.
//
// Note that CheckFiles will not open any files, so Create may still fail when
// CheckFiles is successful due to I/O errors and reported size differences.
func CheckFiles(files []File) (CheckedFiles, error) {
	cf, _, _ := checkFiles(files)
	return cf, cf.Err()
}

// parseGoVers extracts the Go version specified in the given go.mod file.
// It returns an empty string if the version is not found or if an error
// occurs during file parsing.
//
// The version string is in Go toolchain name syntax, prefixed with "go".
// Examples: "go1.21", "go1.22rc2", "go1.23.0"
func parseGoVers(file string, data []byte) string {
	mfile, err := modfile.ParseLax(file, data, nil)
	if err != nil || mfile.Go == nil {
		return ""
	}
	return "go" + mfile.Go.Version
}

// checkFiles implements CheckFiles and also returns lists of valid files and
// their sizes, corresponding to cf.Valid. It omits files in submodules, files
// in vendored packages, symlinked files, and various other unwanted files.
//
// The lists returned are used in Create to avoid repeated calls to File.Lstat.
func checkFiles(files []File) (cf CheckedFiles, validFiles []File, validSizes []int64) {
	errPaths := make(map[string]struct{})
	addError := func(path string, omitted bool, err error) {
		if _, ok := errPaths[path]; ok {
			return
		}
		errPaths[path] = struct{}{}
		fe := FileError{Path: path, Err: err}
		if omitted {
			cf.Omitted = append(cf.Omitted, fe)
		} else {
			cf.Invalid = append(cf.Invalid, fe)
		}
	}

	// Find directories containing go.mod files (other than the root).
	// Files in these directories will be omitted.
	// These directories will not be included in the output zip.
	haveGoMod := make(map[string]bool)
	var vers string
	for _, f := range files {
		p := f.Path()
		dir, base := path.Split(p)
		if strings.EqualFold(base, "go.mod") {
			info, err := f.Lstat()
			if err != nil {
				addError(p, false, err)
				continue
			}
			if !info.Mode().IsRegular() {
				continue
			}
			haveGoMod[dir] = true
			// Extract the Go language version from the root "go.mod" file.
			// This ensures we correctly interpret Go version-specific file omissions.
			// We use f.Open() to handle potential custom Open() implementations
			// that the underlying File type might have.
			if base == "go.mod" && dir == "" {
				if file, err := f.Open(); err == nil {
					if data, err := io.ReadAll(file); err == nil {
						vers = version.Lang(parseGoVers("go.mod", data))
					}
					file.Close()
				}
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

	collisions := make(collisionChecker)
	maxSize := int64(MaxZipFile)
	for _, f := range files {
		p := f.Path()
		if p != path.Clean(p) {
			addError(p, false, errPathNotClean)
			continue
		}
		if path.IsAbs(p) {
			addError(p, false, errPathNotRelative)
			continue
		}
		if isVendoredPackage(p, vers) {
			// Skip files in vendored packages.
			addError(p, true, errVendored)
			continue
		}
		if inSubmodule(p) {
			// Skip submodule files.
			addError(p, true, errSubmoduleFile)
			continue
		}
		if p == ".hg_archival.txt" {
			// Inserted by hg archive.
			// The go command drops this regardless of the VCS being used.
			addError(p, true, errHgArchivalTxt)
			continue
		}
		if err := module.CheckFilePath(p); err != nil {
			addError(p, false, err)
			continue
		}
		if strings.ToLower(p) == "go.mod" && p != "go.mod" {
			addError(p, false, errGoModCase)
			continue
		}
		info, err := f.Lstat()
		if err != nil {
			addError(p, false, err)
			continue
		}
		if err := collisions.check(p, info.IsDir()); err != nil {
			addError(p, false, err)
			continue
		}
		if info.Mode()&os.ModeType == os.ModeSymlink {
			// Skip symbolic links (golang.org/issue/27093).
			addError(p, true, errSymlink)
			continue
		}
		if !info.Mode().IsRegular() {
			addError(p, true, errNotRegular)
			continue
		}
		size := info.Size()
		if size >= 0 && size <= maxSize {
			maxSize -= size
		} else if cf.SizeError == nil {
			cf.SizeError = fmt.Errorf("module source tree too large (max size is %d bytes)", MaxZipFile)
		}
		if p == "go.mod" && size > MaxGoMod {
			addError(p, false, errGoModSize)
			continue
		}
		if p == "LICENSE" && size > MaxLICENSE {
			addError(p, false, errLICENSESize)
			continue
		}

		cf.Valid = append(cf.Valid, p)
		validFiles = append(validFiles, f)
		validSizes = append(validSizes, info.Size())
	}

	return cf, validFiles, validSizes
}

// CheckDir reports whether the files in dir satisfy the name and size
// constraints listed in the package documentation. The returned [CheckedFiles]
// record contains lists of valid, invalid, and omitted files. If a directory is
// omitted (for example, a nested module or vendor directory), it will appear in
// the omitted list, but its files won't be listed.
//
// CheckDir returns an error if it encounters an I/O error or if the returned
// [CheckedFiles] does not describe a valid module zip file (according to
// [CheckedFiles.Err]). The returned [CheckedFiles] is still populated when such
// an error is returned.
//
// Note that CheckDir will not open any files, so [CreateFromDir] may still fail
// when CheckDir is successful due to I/O errors.
func CheckDir(dir string) (CheckedFiles, error) {
	// List files (as CreateFromDir would) and check which ones are omitted
	// or invalid.
	files, omitted, err := listFilesInDir(dir)
	if err != nil {
		return CheckedFiles{}, err
	}
	cf, cfErr := CheckFiles(files)
	_ = cfErr // ignore this error; we'll generate our own after rewriting paths.

	// Replace all paths with file system paths.
	// Paths returned by CheckFiles will be slash-separated paths relative to dir.
	// That's probably not appropriate for error messages.
	for i := range cf.Valid {
		cf.Valid[i] = filepath.Join(dir, cf.Valid[i])
	}
	cf.Omitted = append(cf.Omitted, omitted...)
	for i := range cf.Omitted {
		cf.Omitted[i].Path = filepath.Join(dir, cf.Omitted[i].Path)
	}
	for i := range cf.Invalid {
		cf.Invalid[i].Path = filepath.Join(dir, cf.Invalid[i].Path)
	}
	return cf, cf.Err()
}

// CheckZip reports whether the files contained in a zip file satisfy the name
// and size constraints listed in the package documentation.
//
// CheckZip returns an error if the returned [CheckedFiles] does not describe
// a valid module zip file (according to [CheckedFiles.Err]). The returned
// CheckedFiles is still populated when an error is returned. CheckZip will
// also return an error if the module path or version is malformed or if it
// encounters an error reading the zip file.
//
// Note that CheckZip does not read individual files, so [Unzip] may still fail
// when CheckZip is successful due to I/O errors.
func CheckZip(m module.Version, zipFile string) (CheckedFiles, error) {
	f, err := os.Open(zipFile)
	if err != nil {
		return CheckedFiles{}, err
	}
	defer f.Close()
	_, cf, err := checkZip(m, f)
	return cf, err
}

// checkZip implements checkZip and also returns the *zip.Reader. This is
// used in Unzip to avoid redundant I/O.
func checkZip(m module.Version, f *os.File) (*zip.Reader, CheckedFiles, error) {
	// Make sure the module path and version are valid.
	if vers := module.CanonicalVersion(m.Version); vers != m.Version {
		return nil, CheckedFiles{}, fmt.Errorf("version %q is not canonical (should be %q)", m.Version, vers)
	}
	if err := module.Check(m.Path, m.Version); err != nil {
		return nil, CheckedFiles{}, err
	}

	// Check the total file size.
	info, err := f.Stat()
	if err != nil {
		return nil, CheckedFiles{}, err
	}
	zipSize := info.Size()
	if zipSize > MaxZipFile {
		cf := CheckedFiles{SizeError: fmt.Errorf("module zip file is too large (%d bytes; limit is %d bytes)", zipSize, MaxZipFile)}
		return nil, cf, cf.Err()
	}

	// Check for valid file names, collisions.
	var cf CheckedFiles
	addError := func(zf *zip.File, err error) {
		cf.Invalid = append(cf.Invalid, FileError{Path: zf.Name, Err: err})
	}
	z, err := zip.NewReader(f, zipSize)
	if err != nil {
		return nil, CheckedFiles{}, err
	}
	prefix := fmt.Sprintf("%s@%s/", m.Path, m.Version)
	collisions := make(collisionChecker)
	var size int64
	for _, zf := range z.File {
		if !strings.HasPrefix(zf.Name, prefix) {
			addError(zf, fmt.Errorf("path does not have prefix %q", prefix))
			continue
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
			addError(zf, errPathNotClean)
			continue
		}
		if err := module.CheckFilePath(name); err != nil {
			addError(zf, err)
			continue
		}
		if err := collisions.check(name, isDir); err != nil {
			addError(zf, err)
			continue
		}
		if isDir {
			continue
		}
		if base := path.Base(name); strings.EqualFold(base, "go.mod") {
			if base != name {
				addError(zf, fmt.Errorf("go.mod file not in module root directory"))
				continue
			}
			if name != "go.mod" {
				addError(zf, errGoModCase)
				continue
			}
		}
		sz := int64(zf.UncompressedSize64)
		if sz >= 0 && MaxZipFile-size >= sz {
			size += sz
		} else if cf.SizeError == nil {
			cf.SizeError = fmt.Errorf("total uncompressed size of module contents too large (max size is %d bytes)", MaxZipFile)
		}
		if name == "go.mod" && sz > MaxGoMod {
			addError(zf, fmt.Errorf("go.mod file too large (max size is %d bytes)", MaxGoMod))
			continue
		}
		if name == "LICENSE" && sz > MaxLICENSE {
			addError(zf, fmt.Errorf("LICENSE file too large (max size is %d bytes)", MaxLICENSE))
			continue
		}
		cf.Valid = append(cf.Valid, zf.Name)
	}

	return z, cf, cf.Err()
}

// Create builds a zip archive for module m from an abstract list of files
// and writes it to w.
//
// Create verifies the restrictions described in the package documentation
// and should not produce an archive that [Unzip] cannot extract. Create does not
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

	// Check whether files are valid, not valid, or should be omitted.
	// Also check that the valid files don't exceed the maximum size.
	cf, validFiles, validSizes := checkFiles(files)
	if err := cf.Err(); err != nil {
		return err
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

	for i, f := range validFiles {
		p := f.Path()
		size := validSizes[i]
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
// documentation and should not produce an archive that [Unzip] cannot extract.
// CreateFromDir does not include files in the output archive if they don't
// belong in the module zip. In particular, CreateFromDir will not include
// files in modules found in subdirectories, most files in vendor directories,
// or irregular files (such as symbolic links) in the output archive.
// Additionally, unlike [Create], CreateFromDir will not include directories
// named ".bzr", ".git", ".hg", or ".svn".
func CreateFromDir(w io.Writer, m module.Version, dir string) (err error) {
	defer func() {
		if zerr, ok := err.(*zipError); ok {
			zerr.path = dir
		} else if err != nil {
			err = &zipError{verb: "create zip from directory", path: dir, err: err}
		}
	}()

	files, _, err := listFilesInDir(dir)
	if err != nil {
		return err
	}

	return Create(w, m, files)
}

// CreateFromVCS creates a module zip file for module m from the contents of a
// VCS repository stored locally. The zip content is written to w.
//
// repoRoot must be an absolute path to the base of the repository, such as
// "/Users/some-user/some-repo". If the repository is a Git repository,
// this path is expected to point to its worktree: it can't be a bare git
// repo.
//
// revision is the revision of the repository to create the zip from. Examples
// include HEAD or SHA sums for git repositories.
//
// subdir must be the relative path from the base of the repository, such as
// "sub/dir". To create a zip from the base of the repository, pass an empty
// string.
//
// If CreateFromVCS returns [UnrecognizedVCSError], consider falling back to
// [CreateFromDir].
func CreateFromVCS(w io.Writer, m module.Version, repoRoot, revision, subdir string) (err error) {
	defer func() {
		if zerr, ok := err.(*zipError); ok {
			zerr.path = repoRoot
		} else if err != nil {
			err = &zipError{verb: "create zip from version control system", path: repoRoot, err: err}
		}
	}()

	var filesToCreate []File

	switch {
	case isGitRepo(repoRoot):
		files, err := filesInGitRepo(repoRoot, revision, subdir)
		if err != nil {
			return err
		}

		filesToCreate = files
	default:
		return &UnrecognizedVCSError{RepoRoot: repoRoot}
	}

	return Create(w, m, filesToCreate)
}

// UnrecognizedVCSError indicates that no recognized version control system was
// found in the given directory.
type UnrecognizedVCSError struct {
	RepoRoot string
}

func (e *UnrecognizedVCSError) Error() string {
	return fmt.Sprintf("could not find a recognized version control system at %q", e.RepoRoot)
}

// filesInGitRepo filters out any files that are git ignored in the directory.
func filesInGitRepo(dir, rev, subdir string) ([]File, error) {
	stderr := bytes.Buffer{}
	stdout := bytes.Buffer{}

	// Incredibly, git produces different archives depending on whether
	// it is running on a Windows system or not, in an attempt to normalize
	// text file line endings. Setting -c core.autocrlf=input means only
	// translate files on the way into the repo, not on the way out (archive).
	// The -c core.eol=lf should be unnecessary but set it anyway.
	//
	// Note: We use git archive to understand which files are actually included,
	// ignoring things like .gitignore'd files. We could also use other
	// techniques like git ls-files, but this approach most closely matches what
	// the Go command does, which is beneficial.
	//
	// Note: some of this code copied from https://go.googlesource.com/go/+/refs/tags/go1.16.5/src/cmd/go/internal/modfetch/codehost/git.go#826.
	cmd := exec.Command("git", "-c", "core.autocrlf=input", "-c", "core.eol=lf", "archive", "--format=zip", rev)
	if subdir != "" {
		cmd.Args = append(cmd.Args, subdir)
	}
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "PWD="+dir)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("error running `git archive`: %w, %s", err, stderr.String())
	}

	rawReader := bytes.NewReader(stdout.Bytes())
	zipReader, err := zip.NewReader(rawReader, int64(stdout.Len()))
	if err != nil {
		return nil, err
	}

	haveLICENSE := false
	var fs []File
	for _, zf := range zipReader.File {
		if !strings.HasPrefix(zf.Name, subdir) || strings.HasSuffix(zf.Name, "/") {
			continue
		}

		n := strings.TrimPrefix(zf.Name, subdir)
		if n == "" {
			continue
		}
		n = strings.TrimPrefix(n, "/")

		fs = append(fs, zipFile{
			name: n,
			f:    zf,
		})
		if n == "LICENSE" {
			haveLICENSE = true
		}
	}

	if !haveLICENSE && subdir != "" {
		// Note: this method of extracting the license from the root copied from
		// https://go.googlesource.com/go/+/refs/tags/go1.20.4/src/cmd/go/internal/modfetch/coderepo.go#1118
		// https://go.googlesource.com/go/+/refs/tags/go1.20.4/src/cmd/go/internal/modfetch/codehost/git.go#657
		cmd := exec.Command("git", "cat-file", "blob", rev+":LICENSE")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "PWD="+dir)
		stdout := bytes.Buffer{}
		cmd.Stdout = &stdout
		if err := cmd.Run(); err == nil {
			fs = append(fs, dataFile{name: "LICENSE", data: stdout.Bytes()})
		}
	}

	return fs, nil
}

// isGitRepo reports whether the given directory is a git repo.
func isGitRepo(dir string) bool {
	stdout := &bytes.Buffer{}
	cmd := exec.Command("git", "rev-parse", "--git-dir")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "PWD="+dir)
	cmd.Stdout = stdout
	if err := cmd.Run(); err != nil {
		return false
	}
	gitDir := strings.TrimSpace(stdout.String())
	if !filepath.IsAbs(gitDir) {
		gitDir = filepath.Join(dir, gitDir)
	}
	wantDir := filepath.Join(dir, ".git")
	return wantDir == gitDir
}

type dirFile struct {
	filePath, slashPath string
	info                os.FileInfo
}

func (f dirFile) Path() string                 { return f.slashPath }
func (f dirFile) Lstat() (os.FileInfo, error)  { return f.info, nil }
func (f dirFile) Open() (io.ReadCloser, error) { return os.Open(f.filePath) }

type zipFile struct {
	name string
	f    *zip.File
}

func (f zipFile) Path() string                 { return f.name }
func (f zipFile) Lstat() (os.FileInfo, error)  { return f.f.FileInfo(), nil }
func (f zipFile) Open() (io.ReadCloser, error) { return f.f.Open() }

type dataFile struct {
	name string
	data []byte
}

func (f dataFile) Path() string                 { return f.name }
func (f dataFile) Lstat() (os.FileInfo, error)  { return dataFileInfo{f}, nil }
func (f dataFile) Open() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(f.data)), nil }

type dataFileInfo struct {
	f dataFile
}

func (fi dataFileInfo) Name() string       { return path.Base(fi.f.name) }
func (fi dataFileInfo) Size() int64        { return int64(len(fi.f.data)) }
func (fi dataFileInfo) Mode() os.FileMode  { return 0644 }
func (fi dataFileInfo) ModTime() time.Time { return time.Time{} }
func (fi dataFileInfo) IsDir() bool        { return false }
func (fi dataFileInfo) Sys() interface{}   { return nil }

// isVendoredPackage attempts to report whether the given filename is contained
// in a package whose import path contains (but does not end with) the component
// "vendor".
//
// The 'vers' parameter specifies the Go version declared in the module's
// go.mod file and must be a valid Go version according to the
// go/version.IsValid function.
// Vendoring behavior has evolved across Go versions, so this function adapts
// its logic accordingly.
func isVendoredPackage(name string, vers string) bool {
	// vendor/modules.txt is a vendored package but was included in 1.23 and earlier.
	// Remove vendor/modules.txt only for 1.24 and beyond to preserve older checksums.
	if version.Compare(vers, "go1.24") >= 0 && name == "vendor/modules.txt" {
		return true
	}
	var i int
	if strings.HasPrefix(name, "vendor/") {
		i += len("vendor/")
	} else if j := strings.Index(name, "/vendor/"); j >= 0 {
		// Calculate the correct starting position within the import path
		// to determine if a package is vendored.
		//
		// Due to a bug in Go versions before 1.24
		// (see https://golang.org/issue/37397), the "/vendor/" prefix within
		// a package path was not always correctly interpreted.
		//
		// This bug affected how vendored packages were identified in cases like:
		//
		//   - "pkg/vendor/vendor.go"   (incorrectly identified as vendored in pre-1.24)
		//   - "pkg/vendor/foo/foo.go" (correctly identified as vendored)
		//
		// To correct this, in Go 1.24 and later, we skip the entire "/vendor/" prefix
		// when it's part of a nested package path (as in the first example above).
		// In earlier versions, we only skipped the length of "/vendor/", leading
		// to the incorrect behavior.
		if version.Compare(vers, "go1.24") >= 0 {
			i = j + len("/vendor/")
		} else {
			i += len("/vendor/")
		}
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

	// Check that the directory is empty. Don't create it yet in case there's
	// an error reading the zip.
	if files, _ := os.ReadDir(dir); len(files) > 0 {
		return fmt.Errorf("target directory %v exists and is not empty", dir)
	}

	// Open the zip and check that it satisfies all restrictions.
	f, err := os.Open(zipFile)
	if err != nil {
		return err
	}
	defer f.Close()
	z, cf, err := checkZip(m, f)
	if err != nil {
		return err
	}
	if err := cf.Err(); err != nil {
		return err
	}

	// Unzip, enforcing sizes declared in the zip file.
	prefix := fmt.Sprintf("%s@%s/", m.Path, m.Version)
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

// listFilesInDir walks the directory tree rooted at dir and returns a list of
// files, as well as a list of directories and files that were skipped (for
// example, nested modules and symbolic links).
func listFilesInDir(dir string) (files []File, omitted []FileError, err error) {
	// Extract the Go language version from the root "go.mod" file.
	// This ensures we correctly interpret Go version-specific file omissions.
	var vers string
	if data, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil {
		vers = version.Lang(parseGoVers("go.mod", data))
	}
	err = filepath.Walk(dir, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		relPath, err := filepath.Rel(dir, filePath)
		if err != nil {
			return err
		}
		slashPath := filepath.ToSlash(relPath)

		// Skip some subdirectories inside vendor.
		// We would like Create and CreateFromDir to produce the same result
		// for a set of files, whether expressed as a directory tree or zip.
		if isVendoredPackage(slashPath, vers) {
			omitted = append(omitted, FileError{Path: slashPath, Err: errVendored})
			return nil
		}

		if info.IsDir() {
			if filePath == dir {
				// Don't skip the top-level directory.
				return nil
			}

			// Skip VCS directories.
			// fossil repos are regular files with arbitrary names, so we don't try
			// to exclude them.
			switch filepath.Base(filePath) {
			case ".bzr", ".git", ".hg", ".svn":
				omitted = append(omitted, FileError{Path: slashPath, Err: errVCS})
				return filepath.SkipDir
			}

			// Skip submodules (directories containing go.mod files).
			if goModInfo, err := os.Lstat(filepath.Join(filePath, "go.mod")); err == nil && !goModInfo.IsDir() {
				omitted = append(omitted, FileError{Path: slashPath, Err: errSubmoduleDir})
				return filepath.SkipDir
			}
			return nil
		}

		// Skip irregular files and files in vendor directories.
		// Irregular files are ignored. They're typically symbolic links.
		if !info.Mode().IsRegular() {
			omitted = append(omitted, FileError{Path: slashPath, Err: errNotRegular})
			return nil
		}

		files = append(files, dirFile{
			filePath:  filePath,
			slashPath: slashPath,
			info:      info,
		})
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return files, omitted, nil
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
//
//	strings.EqualFold(s, t) iff strToFold(s) == strToFold(t)
//
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
