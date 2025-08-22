// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package os

import (
	"errors"
	"internal/filepathlite"
	"internal/stringslite"
	"internal/syscall/windows"
	"runtime"
	"syscall"
	"unsafe"
)

// rootCleanPath uses GetFullPathName to perform lexical path cleaning.
//
// On Windows, file names are lexically cleaned at the start of a file operation.
// For example, on Windows the path `a\..\b` is exactly equivalent to `b` alone,
// even if `a` does not exist or is not a directory.
//
// We use the Windows API function GetFullPathName to perform this cleaning.
// We could do this ourselves, but there are a number of subtle behaviors here,
// and deferring to the OS maintains consistency.
// (For example, `a\.\` cleans to `a\`.)
//
// GetFullPathName operates on absolute paths, and our input path is relative.
// We make the path absolute by prepending a fixed prefix of \\?\?\.
//
// We want to detect paths which use .. components to escape the root.
// We do this by ensuring the cleaned path still begins with \\?\?\.
// We catch the corner case of a path which includes a ..\?\. component
// by rejecting any input paths which contain a ?, which is not a valid character
// in a Windows filename.
func rootCleanPath(s string, prefix, suffix []string) (string, error) {
	// Reject paths which include a ? component (see above).
	if stringslite.IndexByte(s, '?') >= 0 {
		return "", windows.ERROR_INVALID_NAME
	}

	const fixedPrefix = `\\?\?`
	buf := []byte(fixedPrefix)
	for _, p := range prefix {
		buf = append(buf, '\\')
		buf = append(buf, []byte(p)...)
	}
	buf = append(buf, '\\')
	buf = append(buf, []byte(s)...)
	for _, p := range suffix {
		buf = append(buf, '\\')
		buf = append(buf, []byte(p)...)
	}
	s = string(buf)

	s, err := syscall.FullPath(s)
	if err != nil {
		return "", err
	}

	s, ok := stringslite.CutPrefix(s, fixedPrefix)
	if !ok {
		return "", errPathEscapes
	}
	s = stringslite.TrimPrefix(s, `\`)
	if s == "" {
		s = "."
	}

	if !filepathlite.IsLocal(s) {
		return "", errPathEscapes
	}

	return s, nil
}

type sysfdType = syscall.Handle

// openRootNolog is OpenRoot.
func openRootNolog(name string) (*Root, error) {
	if name == "" {
		return nil, &PathError{Op: "open", Path: name, Err: syscall.ENOENT}
	}
	path := fixLongPath(name)
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, &PathError{Op: "open", Path: name, Err: err}
	}
	return newRoot(fd, name)
}

// newRoot returns a new Root.
// If fd is not a directory, it closes it and returns an error.
func newRoot(fd syscall.Handle, name string) (*Root, error) {
	// Check that this is a directory.
	//
	// If we get any errors here, ignore them; worst case we create a Root
	// which returns errors when you try to use it.
	var fi syscall.ByHandleFileInformation
	err := syscall.GetFileInformationByHandle(fd, &fi)
	if err == nil && fi.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY == 0 {
		syscall.CloseHandle(fd)
		return nil, &PathError{Op: "open", Path: name, Err: errors.New("not a directory")}
	}

	r := &Root{&root{
		fd:   fd,
		name: name,
	}}
	runtime.SetFinalizer(r.root, (*root).Close)
	return r, nil
}

// openRootInRoot is Root.OpenRoot.
func openRootInRoot(r *Root, name string) (*Root, error) {
	fd, err := doInRoot(r, name, rootOpenDir)
	if err != nil {
		return nil, &PathError{Op: "openat", Path: name, Err: err}
	}
	return newRoot(fd, joinPath(r.Name(), name))
}

// rootOpenFileNolog is Root.OpenFile.
func rootOpenFileNolog(root *Root, name string, flag int, perm FileMode) (*File, error) {
	fd, err := doInRoot(root, name, func(parent syscall.Handle, name string) (syscall.Handle, error) {
		return openat(parent, name, flag, perm)
	})
	if err != nil {
		return nil, &PathError{Op: "openat", Path: name, Err: err}
	}
	return newFile(fd, joinPath(root.Name(), name), "file"), nil
}

func openat(dirfd syscall.Handle, name string, flag int, perm FileMode) (syscall.Handle, error) {
	h, err := windows.Openat(dirfd, name, flag|syscall.O_CLOEXEC|windows.O_NOFOLLOW_ANY, syscallMode(perm))
	if err == syscall.ELOOP || err == syscall.ENOTDIR {
		if link, err := readReparseLinkAt(dirfd, name); err == nil {
			return syscall.InvalidHandle, errSymlink(link)
		}
	}
	return h, err
}

func readReparseLinkAt(dirfd syscall.Handle, name string) (string, error) {
	objectName, err := windows.NewNTUnicodeString(name)
	if err != nil {
		return "", err
	}
	objAttrs := &windows.OBJECT_ATTRIBUTES{
		ObjectName: objectName,
	}
	if dirfd != syscall.InvalidHandle {
		objAttrs.RootDirectory = dirfd
	}
	objAttrs.Length = uint32(unsafe.Sizeof(*objAttrs))
	var h syscall.Handle
	err = windows.NtCreateFile(
		&h,
		windows.FILE_GENERIC_READ,
		objAttrs,
		&windows.IO_STATUS_BLOCK{},
		nil,
		uint32(syscall.FILE_ATTRIBUTE_NORMAL),
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		windows.FILE_OPEN,
		windows.FILE_SYNCHRONOUS_IO_NONALERT|windows.FILE_OPEN_REPARSE_POINT,
		0,
		0,
	)
	if err != nil {
		return "", err
	}
	defer syscall.CloseHandle(h)
	return readReparseLinkHandle(h)
}

func rootOpenDir(parent syscall.Handle, name string) (syscall.Handle, error) {
	h, err := openat(parent, name, syscall.O_RDONLY|syscall.O_CLOEXEC|windows.O_DIRECTORY, 0)
	if err == syscall.ERROR_FILE_NOT_FOUND {
		// Windows returns:
		//   - ERROR_PATH_NOT_FOUND if any path compoenent before the leaf
		//     does not exist or is not a directory.
		//   - ERROR_FILE_NOT_FOUND if the leaf does not exist.
		//
		// This differs from Unix behavior, which is:
		//   - ENOENT if any path component does not exist, including the leaf.
		//   - ENOTDIR if any path component before the leaf is not a directory.
		//
		// We map syscall.ENOENT to ERROR_FILE_NOT_FOUND and syscall.ENOTDIR
		// to ERROR_PATH_NOT_FOUND, but the Windows errors don't quite match.
		//
		// For consistency with os.Open, convert ERROR_FILE_NOT_FOUND here into
		// ERROR_PATH_NOT_FOUND, since we're opening a non-leaf path component.
		err = syscall.ERROR_PATH_NOT_FOUND
	}
	return h, err
}

func rootStat(r *Root, name string, lstat bool) (FileInfo, error) {
	if len(name) > 0 && IsPathSeparator(name[len(name)-1]) {
		// When a filename ends with a path separator,
		// Lstat behaves like Stat.
		//
		// This behavior is not based on a principled decision here,
		// merely the empirical evidence that Lstat behaves this way.
		lstat = false
	}
	fi, err := doInRoot(r, name, func(parent syscall.Handle, n string) (FileInfo, error) {
		fd, err := openat(parent, n, windows.O_OPEN_REPARSE, 0)
		if err != nil {
			return nil, err
		}
		defer syscall.CloseHandle(fd)
		fi, err := statHandle(name, fd)
		if err != nil {
			return nil, err
		}
		if !lstat && fi.(*fileStat).isReparseTagNameSurrogate() {
			link, err := readReparseLinkHandle(fd)
			if err != nil {
				return nil, err
			}
			return nil, errSymlink(link)
		}
		return fi, nil
	})
	if err != nil {
		return nil, &PathError{Op: "statat", Path: name, Err: err}
	}
	return fi, nil
}

func mkdirat(dirfd syscall.Handle, name string, perm FileMode) error {
	return windows.Mkdirat(dirfd, name, syscallMode(perm))
}

func removeat(dirfd syscall.Handle, name string) error {
	return windows.Deleteat(dirfd, name)
}
