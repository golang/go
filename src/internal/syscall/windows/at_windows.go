// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"syscall"
)

// Openat flags not supported by syscall.Open.
//
// These are invented values.
//
// When adding a new flag here, add an unexported version to
// the set of invented O_ values in syscall/types_windows.go
// to avoid overlap.
const (
	O_DIRECTORY    = 0x100000   // target must be a directory
	O_NOFOLLOW_ANY = 0x20000000 // disallow symlinks anywhere in the path
)

func Openat(dirfd syscall.Handle, name string, flag int, perm uint32) (_ syscall.Handle, e1 error) {
	if len(name) == 0 {
		return syscall.InvalidHandle, syscall.ERROR_FILE_NOT_FOUND
	}

	var access, options uint32
	switch flag & (syscall.O_RDONLY | syscall.O_WRONLY | syscall.O_RDWR) {
	case syscall.O_RDONLY:
		// FILE_GENERIC_READ includes FILE_LIST_DIRECTORY.
		access = FILE_GENERIC_READ
	case syscall.O_WRONLY:
		access = FILE_GENERIC_WRITE
		options |= FILE_NON_DIRECTORY_FILE
	case syscall.O_RDWR:
		access = FILE_GENERIC_READ | FILE_GENERIC_WRITE
		options |= FILE_NON_DIRECTORY_FILE
	}
	if flag&syscall.O_CREAT != 0 {
		access |= FILE_GENERIC_WRITE
	}
	if flag&syscall.O_APPEND != 0 {
		access |= FILE_APPEND_DATA
		// Remove GENERIC_WRITE access unless O_TRUNC is set,
		// in which case we need it to truncate the file.
		if flag&syscall.O_TRUNC == 0 {
			access &^= FILE_WRITE_DATA
		}
	}
	if flag&O_DIRECTORY != 0 {
		options |= FILE_DIRECTORY_FILE
		access |= FILE_LIST_DIRECTORY
	}
	if flag&syscall.O_SYNC != 0 {
		options |= FILE_WRITE_THROUGH
	}
	// Allow File.Stat.
	access |= STANDARD_RIGHTS_READ | FILE_READ_ATTRIBUTES | FILE_READ_EA

	objAttrs := &OBJECT_ATTRIBUTES{}
	if flag&O_NOFOLLOW_ANY != 0 {
		objAttrs.Attributes |= OBJ_DONT_REPARSE
	}
	if flag&syscall.O_CLOEXEC == 0 {
		objAttrs.Attributes |= OBJ_INHERIT
	}
	if err := objAttrs.init(dirfd, name); err != nil {
		return syscall.InvalidHandle, err
	}

	// We don't use FILE_OVERWRITE/FILE_OVERWRITE_IF, because when opening
	// a file with FILE_ATTRIBUTE_READONLY these will replace an existing
	// file with a new, read-only one.
	//
	// Instead, we ftruncate the file after opening when O_TRUNC is set.
	var disposition uint32
	switch {
	case flag&(syscall.O_CREAT|syscall.O_EXCL) == (syscall.O_CREAT | syscall.O_EXCL):
		disposition = FILE_CREATE
	case flag&syscall.O_CREAT == syscall.O_CREAT:
		disposition = FILE_OPEN_IF
	default:
		disposition = FILE_OPEN
	}

	fileAttrs := uint32(FILE_ATTRIBUTE_NORMAL)
	if perm&syscall.S_IWRITE == 0 {
		fileAttrs = FILE_ATTRIBUTE_READONLY
	}

	var h syscall.Handle
	err := NtCreateFile(
		&h,
		SYNCHRONIZE|access,
		objAttrs,
		&IO_STATUS_BLOCK{},
		nil,
		fileAttrs,
		FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
		disposition,
		FILE_SYNCHRONOUS_IO_NONALERT|options,
		0,
		0,
	)
	if err != nil {
		return h, ntCreateFileError(err, flag)
	}

	if flag&syscall.O_TRUNC != 0 {
		err = syscall.Ftruncate(h, 0)
		if err != nil {
			syscall.CloseHandle(h)
			return syscall.InvalidHandle, err
		}
	}

	return h, nil
}

// ntCreateFileError maps error returns from NTCreateFile to user-visible errors.
func ntCreateFileError(err error, flag int) error {
	s, ok := err.(NTStatus)
	if !ok {
		// Shouldn't really be possible, NtCreateFile always returns NTStatus.
		return err
	}
	switch s {
	case STATUS_REPARSE_POINT_ENCOUNTERED:
		return syscall.ELOOP
	case STATUS_NOT_A_DIRECTORY:
		// ENOTDIR is the errno returned by open when O_DIRECTORY is specified
		// and the target is not a directory.
		//
		// NtCreateFile can return STATUS_NOT_A_DIRECTORY under other circumstances,
		// such as when opening "file/" where "file" is not a directory.
		// (This might be Windows version dependent.)
		//
		// Only map STATUS_NOT_A_DIRECTORY to ENOTDIR when O_DIRECTORY is specified.
		if flag&O_DIRECTORY != 0 {
			return syscall.ENOTDIR
		}
	case STATUS_FILE_IS_A_DIRECTORY:
		return syscall.EISDIR
	}
	return s.Errno()
}

func Mkdirat(dirfd syscall.Handle, name string, mode uint32) error {
	objAttrs := &OBJECT_ATTRIBUTES{}
	if err := objAttrs.init(dirfd, name); err != nil {
		return err
	}
	var h syscall.Handle
	err := NtCreateFile(
		&h,
		FILE_GENERIC_READ,
		objAttrs,
		&IO_STATUS_BLOCK{},
		nil,
		syscall.FILE_ATTRIBUTE_NORMAL,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		FILE_CREATE,
		FILE_DIRECTORY_FILE,
		0,
		0,
	)
	if err != nil {
		return ntCreateFileError(err, 0)
	}
	syscall.CloseHandle(h)
	return nil
}
