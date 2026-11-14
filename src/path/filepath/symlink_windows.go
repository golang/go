// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import (
	"io/fs"
	"strings"
	"syscall"
)

// isDirReparsePoint reports whether fi, as returned by [os.Lstat],
// describes a reparse point for a directory that is reported as
// neither a directory nor a symbolic link, such as a directory
// junction or another mount point. EvalSymlinks does not evaluate
// these, but a path may go through them.
func isDirReparsePoint(fi fs.FileInfo) bool {
	d, ok := fi.Sys().(*syscall.Win32FileAttributeData)
	return ok && d.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT != 0 &&
		d.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY != 0
}

// normVolumeName is like VolumeName, but makes drive letter upper case.
// result of EvalSymlinks must be unique, so we have
// EvalSymlinks(`c:\a`) == EvalSymlinks(`C:\a`).
func normVolumeName(path string) string {
	volume := VolumeName(path)

	if len(volume) > 2 { // isUNC
		return volume
	}

	return strings.ToUpper(volume)
}

// normBase returns the last element of path with correct case.
func normBase(path string) (string, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return "", err
	}

	var data syscall.Win32finddata

	h, err := syscall.FindFirstFile(p, &data)
	if err != nil {
		return "", err
	}
	syscall.FindClose(h)

	return syscall.UTF16ToString(data.FileName[:]), nil
}

// baseIsDotDot reports whether the last element of path is "..".
// The given path should be 'Clean'-ed in advance.
func baseIsDotDot(path string) bool {
	i := strings.LastIndexByte(path, Separator)
	return path[i+1:] == ".."
}

// toNorm returns the normalized path that is guaranteed to be unique.
// It should accept the following formats:
//   - UNC paths                              (e.g \\server\share\foo\bar)
//   - absolute paths                         (e.g C:\foo\bar)
//   - relative paths begin with drive letter (e.g C:foo\bar, C:..\foo\bar, C:.., C:.)
//   - relative paths begin with '\'          (e.g \foo\bar)
//   - relative paths begin without '\'       (e.g foo\bar, ..\foo\bar, .., .)
//
// The returned normalized path will be in the same form (of 5 listed above) as the input path.
// If two paths A and B are indicating the same file with the same format, toNorm(A) should be equal to toNorm(B).
// The normBase parameter should be equal to the normBase func, except for in tests.  See docs on the normBase func.
func toNorm(path string, normBase func(string) (string, error)) (string, error) {
	if path == "" {
		return path, nil
	}

	volume := normVolumeName(path)
	path = path[len(volume):]

	// skip special cases
	if path == "" || path == "." || path == `\` {
		return volume + path, nil
	}

	var normPath string

	for {
		if baseIsDotDot(path) {
			normPath = path + `\` + normPath

			break
		}

		name, err := normBase(volume + path)
		if err != nil {
			return "", err
		}

		normPath = name + `\` + normPath

		i := strings.LastIndexByte(path, Separator)
		if i == -1 {
			break
		}
		if i == 0 { // `\Go` or `C:\Go`
			normPath = `\` + normPath

			break
		}

		path = path[:i]
	}

	normPath = normPath[:len(normPath)-1] // remove trailing '\'

	return volume + normPath, nil
}

func evalSymlinks(path string) (string, error) {
	newpath, err := walkSymlinks(path)
	if err != nil {
		return "", err
	}
	newpath, err = toNorm(newpath, normBase)
	if err != nil {
		return "", err
	}
	return newpath, nil
}
