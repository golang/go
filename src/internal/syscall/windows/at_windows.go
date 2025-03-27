// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"syscall"
	"unsafe"
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
	O_OPEN_REPARSE = 0x40000000 // FILE_OPEN_REPARSE_POINT, used by Lstat
	O_WRITE_ATTRS  = 0x80000000 // FILE_WRITE_ATTRIBUTES, used by Chmod
)

func Openat(dirfd syscall.Handle, name string, flag uint64, perm uint32) (_ syscall.Handle, e1 error) {
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
	default:
		// Stat opens files without requesting read or write permissions,
		// but we still need to request SYNCHRONIZE.
		access = SYNCHRONIZE
	}
	if flag&syscall.O_CREAT != 0 {
		access |= FILE_GENERIC_WRITE
	}
	if flag&syscall.O_APPEND != 0 {
		access |= FILE_APPEND_DATA
		// Remove FILE_WRITE_DATA access unless O_TRUNC is set,
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
	if flag&O_WRITE_ATTRS != 0 {
		access |= FILE_WRITE_ATTRIBUTES
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

	if flag&O_OPEN_REPARSE != 0 {
		options |= FILE_OPEN_REPARSE_POINT
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
		FILE_SYNCHRONOUS_IO_NONALERT|FILE_OPEN_FOR_BACKUP_INTENT|options,
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
func ntCreateFileError(err error, flag uint64) error {
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

func Deleteat(dirfd syscall.Handle, name string) error {
	objAttrs := &OBJECT_ATTRIBUTES{}
	if err := objAttrs.init(dirfd, name); err != nil {
		return err
	}
	var h syscall.Handle
	err := NtOpenFile(
		&h,
		DELETE,
		objAttrs,
		&IO_STATUS_BLOCK{},
		FILE_SHARE_DELETE|FILE_SHARE_READ|FILE_SHARE_WRITE,
		FILE_OPEN_REPARSE_POINT|FILE_OPEN_FOR_BACKUP_INTENT,
	)
	if err != nil {
		return ntCreateFileError(err, 0)
	}
	defer syscall.CloseHandle(h)

	const (
		FileDispositionInformation   = 13
		FileDispositionInformationEx = 64
	)

	// First, attempt to delete the file using POSIX semantics
	// (which permit a file to be deleted while it is still open).
	// This matches the behavior of DeleteFileW.
	err = NtSetInformationFile(
		h,
		&IO_STATUS_BLOCK{},
		uintptr(unsafe.Pointer(&FILE_DISPOSITION_INFORMATION_EX{
			Flags: FILE_DISPOSITION_DELETE |
				FILE_DISPOSITION_FORCE_IMAGE_SECTION_CHECK |
				FILE_DISPOSITION_POSIX_SEMANTICS |
				// This differs from DeleteFileW, but matches os.Remove's
				// behavior on Unix platforms of permitting deletion of
				// read-only files.
				FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE,
		})),
		uint32(unsafe.Sizeof(FILE_DISPOSITION_INFORMATION_EX{})),
		FileDispositionInformationEx,
	)
	switch err {
	case nil:
		return nil
	case STATUS_CANNOT_DELETE, STATUS_DIRECTORY_NOT_EMPTY:
		return err.(NTStatus).Errno()
	}

	// If the prior deletion failed, the filesystem either doesn't support
	// POSIX semantics (for example, FAT), or hasn't implemented
	// FILE_DISPOSITION_INFORMATION_EX.
	//
	// Try again.
	err = NtSetInformationFile(
		h,
		&IO_STATUS_BLOCK{},
		uintptr(unsafe.Pointer(&FILE_DISPOSITION_INFORMATION{
			DeleteFile: true,
		})),
		uint32(unsafe.Sizeof(FILE_DISPOSITION_INFORMATION{})),
		FileDispositionInformation,
	)
	if st, ok := err.(NTStatus); ok {
		return st.Errno()
	}
	return err
}
