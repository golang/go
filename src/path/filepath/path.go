// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package filepath implements utility routines for manipulating filename paths
// in a way compatible with the target operating system-defined file paths.
//
// The filepath package uses either forward slashes or backslashes,
// depending on the operating system. To process paths such as URLs
// that always use forward slashes regardless of the operating
// system, see the [path] package.
package filepath

import (
	"errors"
	"internal/bytealg"
	"internal/filepathlite"
	"io/fs"
	"os"
	"slices"
)

const (
	Separator     = os.PathSeparator
	ListSeparator = os.PathListSeparator
)

// Clean returns the shortest path name equivalent to path
// by purely lexical processing. It applies the following rules
// iteratively until no further processing can be done:
//
//  1. Replace multiple [Separator] elements with a single one.
//  2. Eliminate each . path name element (the current directory).
//  3. Eliminate each inner .. path name element (the parent directory)
//     along with the non-.. element that precedes it.
//  4. Eliminate .. elements that begin a rooted path:
//     that is, replace "/.." by "/" at the beginning of a path,
//     assuming Separator is '/'.
//
// The returned path ends in a slash only if it represents a root directory,
// such as "/" on Unix or `C:\` on Windows.
//
// Finally, any occurrences of slash are replaced by Separator.
//
// If the result of this process is an empty string, Clean
// returns the string ".".
//
// On Windows, Clean does not modify the volume name other than to replace
// occurrences of "/" with `\`.
// For example, Clean("//host/share/../x") returns `\\host\share\x`.
//
// See also Rob Pike, “Lexical File Names in Plan 9 or
// Getting Dot-Dot Right,”
// https://9p.io/sys/doc/lexnames.html
func Clean(path string) string {
	return filepathlite.Clean(path)
}

// IsLocal reports whether path, using lexical analysis only, has all of these properties:
//
//   - is within the subtree rooted at the directory in which path is evaluated
//   - is not an absolute path
//   - is not empty
//   - on Windows, is not a reserved name such as "NUL"
//
// If IsLocal(path) returns true, then
// Join(base, path) will always produce a path contained within base and
// Clean(path) will always produce an unrooted path with no ".." path elements.
//
// IsLocal is a purely lexical operation.
// In particular, it does not account for the effect of any symbolic links
// that may exist in the filesystem.
func IsLocal(path string) bool {
	return filepathlite.IsLocal(path)
}

// Localize converts a slash-separated path into an operating system path.
// The input path must be a valid path as reported by [io/fs.ValidPath].
//
// Localize returns an error if the path cannot be represented by the operating system.
// For example, the path a\b is rejected on Windows, on which \ is a separator
// character and cannot be part of a filename.
//
// The path returned by Localize will always be local, as reported by IsLocal.
func Localize(path string) (string, error) {
	return filepathlite.Localize(path)
}

// ToSlash returns the result of replacing each separator character
// in path with a slash ('/') character. Multiple separators are
// replaced by multiple slashes.
func ToSlash(path string) string {
	return filepathlite.ToSlash(path)
}

// FromSlash returns the result of replacing each slash ('/') character
// in path with a separator character. Multiple slashes are replaced
// by multiple separators.
//
// See also the Localize function, which converts a slash-separated path
// as used by the io/fs package to an operating system path.
func FromSlash(path string) string {
	return filepathlite.FromSlash(path)
}

// SplitList splits a list of paths joined by the OS-specific [ListSeparator],
// usually found in PATH or GOPATH environment variables.
// Unlike strings.Split, SplitList returns an empty slice when passed an empty
// string.
func SplitList(path string) []string {
	return splitList(path)
}

// Split splits path immediately following the final [Separator],
// separating it into a directory and file name component.
// If there is no Separator in path, Split returns an empty dir
// and file set to path.
// The returned values have the property that path = dir+file.
func Split(path string) (dir, file string) {
	return filepathlite.Split(path)
}

// Join joins any number of path elements into a single path,
// separating them with an OS specific [Separator]. Empty elements
// are ignored. The result is Cleaned. However, if the argument
// list is empty or all its elements are empty, Join returns
// an empty string.
// On Windows, the result will only be a UNC path if the first
// non-empty element is a UNC path.
func Join(elem ...string) string {
	return join(elem)
}

// Ext returns the file name extension used by path.
// The extension is the suffix beginning at the final dot
// in the final element of path; it is empty if there is
// no dot.
func Ext(path string) string {
	return filepathlite.Ext(path)
}

// EvalSymlinks returns the path name after the evaluation of any symbolic
// links.
// If path is relative the result will be relative to the current directory,
// unless one of the components is an absolute symbolic link.
// EvalSymlinks calls [Clean] on the result.
func EvalSymlinks(path string) (string, error) {
	return evalSymlinks(path)
}

// IsAbs reports whether the path is absolute.
func IsAbs(path string) bool {
	return filepathlite.IsAbs(path)
}

// Abs returns an absolute representation of path.
// If the path is not absolute it will be joined with the current
// working directory to turn it into an absolute path. The absolute
// path name for a given file is not guaranteed to be unique.
// Abs calls [Clean] on the result.
func Abs(path string) (string, error) {
	return abs(path)
}

func unixAbs(path string) (string, error) {
	if IsAbs(path) {
		return Clean(path), nil
	}
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	return Join(wd, path), nil
}

// Rel returns a relative path that is lexically equivalent to targpath when
// joined to basepath with an intervening separator. That is,
// [Join](basepath, Rel(basepath, targpath)) is equivalent to targpath itself.
// On success, the returned path will always be relative to basepath,
// even if basepath and targpath share no elements.
// An error is returned if targpath can't be made relative to basepath or if
// knowing the current working directory would be necessary to compute it.
// Rel calls [Clean] on the result.
func Rel(basepath, targpath string) (string, error) {
	baseVol := VolumeName(basepath)
	targVol := VolumeName(targpath)
	base := Clean(basepath)
	targ := Clean(targpath)
	if sameWord(targ, base) {
		return ".", nil
	}
	base = base[len(baseVol):]
	targ = targ[len(targVol):]
	if base == "." {
		base = ""
	} else if base == "" && filepathlite.VolumeNameLen(baseVol) > 2 /* isUNC */ {
		// Treat any targetpath matching `\\host\share` basepath as absolute path.
		base = string(Separator)
	}

	// Can't use IsAbs - `\a` and `a` are both relative in Windows.
	baseSlashed := len(base) > 0 && base[0] == Separator
	targSlashed := len(targ) > 0 && targ[0] == Separator
	if baseSlashed != targSlashed || !sameWord(baseVol, targVol) {
		return "", errors.New("Rel: can't make " + targpath + " relative to " + basepath)
	}
	// Position base[b0:bi] and targ[t0:ti] at the first differing elements.
	bl := len(base)
	tl := len(targ)
	var b0, bi, t0, ti int
	for {
		for bi < bl && base[bi] != Separator {
			bi++
		}
		for ti < tl && targ[ti] != Separator {
			ti++
		}
		if !sameWord(targ[t0:ti], base[b0:bi]) {
			break
		}
		if bi < bl {
			bi++
		}
		if ti < tl {
			ti++
		}
		b0 = bi
		t0 = ti
	}
	if base[b0:bi] == ".." {
		return "", errors.New("Rel: can't make " + targpath + " relative to " + basepath)
	}
	if b0 != bl {
		// Base elements left. Must go up before going down.
		seps := bytealg.CountString(base[b0:bl], Separator)
		size := 2 + seps*3
		if tl != t0 {
			size += 1 + tl - t0
		}
		buf := make([]byte, size)
		n := copy(buf, "..")
		for i := 0; i < seps; i++ {
			buf[n] = Separator
			copy(buf[n+1:], "..")
			n += 3
		}
		if t0 != tl {
			buf[n] = Separator
			copy(buf[n+1:], targ[t0:])
		}
		return string(buf), nil
	}
	return targ[t0:], nil
}

// SkipDir is used as a return value from [WalkFunc] to indicate that
// the directory named in the call is to be skipped. It is not returned
// as an error by any function.
var SkipDir error = fs.SkipDir

// SkipAll is used as a return value from [WalkFunc] to indicate that
// all remaining files and directories are to be skipped. It is not returned
// as an error by any function.
var SkipAll error = fs.SkipAll

// WalkFunc is the type of the function called by [Walk] to visit each
// file or directory.
//
// The path argument contains the argument to Walk as a prefix.
// That is, if Walk is called with root argument "dir" and finds a file
// named "a" in that directory, the walk function will be called with
// argument "dir/a".
//
// The directory and file are joined with Join, which may clean the
// directory name: if Walk is called with the root argument "x/../dir"
// and finds a file named "a" in that directory, the walk function will
// be called with argument "dir/a", not "x/../dir/a".
//
// The info argument is the fs.FileInfo for the named path.
//
// The error result returned by the function controls how Walk continues.
// If the function returns the special value [SkipDir], Walk skips the
// current directory (path if info.IsDir() is true, otherwise path's
// parent directory). If the function returns the special value [SkipAll],
// Walk skips all remaining files and directories. Otherwise, if the function
// returns a non-nil error, Walk stops entirely and returns that error.
//
// The err argument reports an error related to path, signaling that Walk
// will not walk into that directory. The function can decide how to
// handle that error; as described earlier, returning the error will
// cause Walk to stop walking the entire tree.
//
// Walk calls the function with a non-nil err argument in two cases.
//
// First, if an [os.Lstat] on the root directory or any directory or file
// in the tree fails, Walk calls the function with path set to that
// directory or file's path, info set to nil, and err set to the error
// from os.Lstat.
//
// Second, if a directory's Readdirnames method fails, Walk calls the
// function with path set to the directory's path, info, set to an
// [fs.FileInfo] describing the directory, and err set to the error from
// Readdirnames.
type WalkFunc func(path string, info fs.FileInfo, err error) error

var lstat = os.Lstat // for testing

// walkDir recursively descends path, calling walkDirFn.
func walkDir(path string, d fs.DirEntry, walkDirFn fs.WalkDirFunc) error {
	if err := walkDirFn(path, d, nil); err != nil || !d.IsDir() {
		if err == SkipDir && d.IsDir() {
			// Successfully skipped directory.
			err = nil
		}
		return err
	}

	dirs, err := os.ReadDir(path)
	if err != nil {
		// Second call, to report ReadDir error.
		err = walkDirFn(path, d, err)
		if err != nil {
			if err == SkipDir && d.IsDir() {
				err = nil
			}
			return err
		}
	}

	for _, d1 := range dirs {
		path1 := Join(path, d1.Name())
		if err := walkDir(path1, d1, walkDirFn); err != nil {
			if err == SkipDir {
				break
			}
			return err
		}
	}
	return nil
}

// walk recursively descends path, calling walkFn.
func walk(path string, info fs.FileInfo, walkFn WalkFunc) error {
	if !info.IsDir() {
		return walkFn(path, info, nil)
	}

	names, err := readDirNames(path)
	err1 := walkFn(path, info, err)
	// If err != nil, walk can't walk into this directory.
	// err1 != nil means walkFn want walk to skip this directory or stop walking.
	// Therefore, if one of err and err1 isn't nil, walk will return.
	if err != nil || err1 != nil {
		// The caller's behavior is controlled by the return value, which is decided
		// by walkFn. walkFn may ignore err and return nil.
		// If walkFn returns SkipDir or SkipAll, it will be handled by the caller.
		// So walk should return whatever walkFn returns.
		return err1
	}

	for _, name := range names {
		filename := Join(path, name)
		fileInfo, err := lstat(filename)
		if err != nil {
			if err := walkFn(filename, fileInfo, err); err != nil && err != SkipDir {
				return err
			}
		} else {
			err = walk(filename, fileInfo, walkFn)
			if err != nil {
				if !fileInfo.IsDir() || err != SkipDir {
					return err
				}
			}
		}
	}
	return nil
}

// WalkDir walks the file tree rooted at root, calling fn for each file or
// directory in the tree, including root.
//
// All errors that arise visiting files and directories are filtered by fn:
// see the [fs.WalkDirFunc] documentation for details.
//
// The files are walked in lexical order, which makes the output deterministic
// but requires WalkDir to read an entire directory into memory before proceeding
// to walk that directory.
//
// WalkDir does not follow symbolic links.
//
// WalkDir calls fn with paths that use the separator character appropriate
// for the operating system. This is unlike [io/fs.WalkDir], which always
// uses slash separated paths.
func WalkDir(root string, fn fs.WalkDirFunc) error {
	info, err := os.Lstat(root)
	if err != nil {
		err = fn(root, nil, err)
	} else {
		err = walkDir(root, fs.FileInfoToDirEntry(info), fn)
	}
	if err == SkipDir || err == SkipAll {
		return nil
	}
	return err
}

// Walk walks the file tree rooted at root, calling fn for each file or
// directory in the tree, including root.
//
// All errors that arise visiting files and directories are filtered by fn:
// see the [WalkFunc] documentation for details.
//
// The files are walked in lexical order, which makes the output deterministic
// but requires Walk to read an entire directory into memory before proceeding
// to walk that directory.
//
// Walk does not follow symbolic links.
//
// Walk is less efficient than [WalkDir], introduced in Go 1.16,
// which avoids calling os.Lstat on every visited file or directory.
func Walk(root string, fn WalkFunc) error {
	info, err := os.Lstat(root)
	if err != nil {
		err = fn(root, nil, err)
	} else {
		err = walk(root, info, fn)
	}
	if err == SkipDir || err == SkipAll {
		return nil
	}
	return err
}

// readDirNames reads the directory named by dirname and returns
// a sorted list of directory entry names.
func readDirNames(dirname string) ([]string, error) {
	f, err := os.Open(dirname)
	if err != nil {
		return nil, err
	}
	names, err := f.Readdirnames(-1)
	f.Close()
	if err != nil {
		return nil, err
	}
	slices.Sort(names)
	return names, nil
}

// Base returns the last element of path.
// Trailing path separators are removed before extracting the last element.
// If the path is empty, Base returns ".".
// If the path consists entirely of separators, Base returns a single separator.
func Base(path string) string {
	return filepathlite.Base(path)
}

// Dir returns all but the last element of path, typically the path's directory.
// After dropping the final element, Dir calls [Clean] on the path and trailing
// slashes are removed.
// If the path is empty, Dir returns ".".
// If the path consists entirely of separators, Dir returns a single separator.
// The returned path does not end in a separator unless it is the root directory.
func Dir(path string) string {
	return filepathlite.Dir(path)
}

// VolumeName returns leading volume name.
// Given "C:\foo\bar" it returns "C:" on Windows.
// Given "\\host\share\foo" it returns "\\host\share".
// On other platforms it returns "".
func VolumeName(path string) string {
	return filepathlite.VolumeName(path)
}
