// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import (
	"errors"
	"internal/syscall/windows"
	"os"
	"strings"
	"syscall"
)

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
//   * UNC paths                              (e.g \\server\share\foo\bar)
//   * absolute paths                         (e.g C:\foo\bar)
//   * relative paths begin with drive letter (e.g C:foo\bar, C:..\foo\bar, C:.., C:.)
//   * relative paths begin with '\'          (e.g \foo\bar)
//   * relative paths begin without '\'       (e.g foo\bar, ..\foo\bar, .., .)
// The returned normalized path will be in the same form (of 5 listed above) as the input path.
// If two paths A and B are indicating the same file with the same format, toNorm(A) should be equal to toNorm(B).
// The normBase parameter should be equal to the normBase func, except for in tests.  See docs on the normBase func.
func toNorm(path string, normBase func(string) (string, error)) (string, error) {
	if path == "" {
		return path, nil
	}

	path = Clean(path)

	volume := normVolumeName(path)
	path = path[len(volume):]

	// skip special cases
	if path == "." || path == `\` {
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

// evalSymlinksUsingGetFinalPathNameByHandle uses Windows
// GetFinalPathNameByHandle API to retrieve the final
// path for the specified file.
func evalSymlinksUsingGetFinalPathNameByHandle(path string) (string, error) {
	err := windows.LoadGetFinalPathNameByHandle()
	if err != nil {
		// we must be using old version of Windows
		return "", err
	}

	if path == "" {
		return path, nil
	}

	// Use Windows I/O manager to dereference the symbolic link, as per
	// https://blogs.msdn.microsoft.com/oldnewthing/20100212-00/?p=14963/
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return "", err
	}
	h, err := syscall.CreateFile(p, 0, 0, nil,
		syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return "", err
	}
	defer syscall.CloseHandle(h)

	buf := make([]uint16, 100)
	for {
		n, err := windows.GetFinalPathNameByHandle(h, &buf[0], uint32(len(buf)), windows.VOLUME_NAME_DOS)
		if err != nil {
			return "", err
		}
		if n < uint32(len(buf)) {
			break
		}
		buf = make([]uint16, n)
	}
	s := syscall.UTF16ToString(buf)
	if len(s) > 4 && s[:4] == `\\?\` {
		s = s[4:]
		if len(s) > 3 && s[:3] == `UNC` {
			// return path like \\server\share\...
			return `\` + s[3:], nil
		}
		return s, nil
	}
	return "", errors.New("GetFinalPathNameByHandle returned unexpected path=" + s)
}

func samefile(path1, path2 string) bool {
	fi1, err := os.Lstat(path1)
	if err != nil {
		return false
	}
	fi2, err := os.Lstat(path2)
	if err != nil {
		return false
	}
	return os.SameFile(fi1, fi2)
}

// walkSymlinks returns slashAfterFilePathError error for paths like
// //path/to/existing_file/ and /path/to/existing_file/. and /path/to/existing_file/..

var slashAfterFilePathError = errors.New("attempting to walk past file path.")

func evalSymlinks(path string) (string, error) {
	newpath, err := walkSymlinks(path)
	if err == slashAfterFilePathError {
		return "", syscall.ENOTDIR
	}
	if err != nil {
		newpath2, err2 := evalSymlinksUsingGetFinalPathNameByHandle(path)
		if err2 == nil {
			return toNorm(newpath2, normBase)
		}
		return "", err
	}
	newpath, err = toNorm(newpath, normBase)
	if err != nil {
		newpath2, err2 := evalSymlinksUsingGetFinalPathNameByHandle(path)
		if err2 == nil {
			return toNorm(newpath2, normBase)
		}
		return "", err
	}
	if strings.ToUpper(newpath) == strings.ToUpper(path) {
		// walkSymlinks did not actually walk any symlinks,
		// so we don't need to try GetFinalPathNameByHandle.
		return newpath, nil
	}
	newpath2, err2 := evalSymlinksUsingGetFinalPathNameByHandle(path)
	if err2 != nil {
		return newpath, nil
	}
	newpath2, err2 = toNorm(newpath2, normBase)
	if err2 != nil {
		return newpath, nil
	}
	if samefile(newpath, newpath2) {
		return newpath, nil
	}
	return newpath2, nil
}
