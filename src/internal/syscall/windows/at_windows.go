// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"runtime"
	"structs"
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
		options |= FILE_OPEN_REPARSE_POINT // don't follow symlinks
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
		nil,
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
	case STATUS_OBJECT_NAME_COLLISION:
		return syscall.EEXIST
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
		nil,
		0,
	)
	if err != nil {
		return ntCreateFileError(err, 0)
	}
	syscall.CloseHandle(h)
	return nil
}

func Deleteat(dirfd syscall.Handle, name string, options uint32) error {
	if name == "." {
		// NtOpenFile's documentation isn't explicit about what happens when deleting ".".
		// Make this an error consistent with that of POSIX.
		return syscall.EINVAL
	}
	objAttrs := &OBJECT_ATTRIBUTES{}
	if err := objAttrs.init(dirfd, name); err != nil {
		return err
	}
	var h syscall.Handle
	err := NtOpenFile(
		&h,
		SYNCHRONIZE|DELETE,
		objAttrs,
		&IO_STATUS_BLOCK{},
		FILE_SHARE_DELETE|FILE_SHARE_READ|FILE_SHARE_WRITE,
		FILE_OPEN_REPARSE_POINT|FILE_OPEN_FOR_BACKUP_INTENT|FILE_SYNCHRONOUS_IO_NONALERT|options,
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
		unsafe.Pointer(&FILE_DISPOSITION_INFORMATION_EX{
			Flags: FILE_DISPOSITION_DELETE |
				FILE_DISPOSITION_FORCE_IMAGE_SECTION_CHECK |
				FILE_DISPOSITION_POSIX_SEMANTICS |
				// This differs from DeleteFileW, but matches os.Remove's
				// behavior on Unix platforms of permitting deletion of
				// read-only files.
				FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE,
		}),
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
		unsafe.Pointer(&FILE_DISPOSITION_INFORMATION{
			DeleteFile: true,
		}),
		uint32(unsafe.Sizeof(FILE_DISPOSITION_INFORMATION{})),
		FileDispositionInformation,
	)
	if st, ok := err.(NTStatus); ok {
		return st.Errno()
	}
	return err
}

func Renameat(olddirfd syscall.Handle, oldpath string, newdirfd syscall.Handle, newpath string) error {
	objAttrs := &OBJECT_ATTRIBUTES{}
	if err := objAttrs.init(olddirfd, oldpath); err != nil {
		return err
	}
	var h syscall.Handle
	err := NtOpenFile(
		&h,
		SYNCHRONIZE|DELETE,
		objAttrs,
		&IO_STATUS_BLOCK{},
		FILE_SHARE_DELETE|FILE_SHARE_READ|FILE_SHARE_WRITE,
		FILE_OPEN_REPARSE_POINT|FILE_OPEN_FOR_BACKUP_INTENT|FILE_SYNCHRONOUS_IO_NONALERT,
	)
	if err != nil {
		return ntCreateFileError(err, 0)
	}
	defer syscall.CloseHandle(h)

	renameInfoEx := FILE_RENAME_INFORMATION_EX{
		Flags: FILE_RENAME_REPLACE_IF_EXISTS |
			FILE_RENAME_POSIX_SEMANTICS,
		RootDirectory: newdirfd,
	}
	p16, err := syscall.UTF16FromString(newpath)
	if err != nil {
		return err
	}
	if len(p16) > len(renameInfoEx.FileName) {
		return syscall.EINVAL
	}
	copy(renameInfoEx.FileName[:], p16)
	renameInfoEx.FileNameLength = uint32((len(p16) - 1) * 2)

	const (
		FileRenameInformation   = 10
		FileRenameInformationEx = 65
	)
	err = NtSetInformationFile(
		h,
		&IO_STATUS_BLOCK{},
		unsafe.Pointer(&renameInfoEx),
		uint32(unsafe.Sizeof(FILE_RENAME_INFORMATION_EX{})),
		FileRenameInformationEx,
	)
	if err == nil {
		return nil
	}

	// If the prior rename failed, the filesystem might not support
	// POSIX semantics (for example, FAT), or might not have implemented
	// FILE_RENAME_INFORMATION_EX.
	//
	// Try again.
	renameInfo := FILE_RENAME_INFORMATION{
		ReplaceIfExists: true,
		RootDirectory:   newdirfd,
	}
	copy(renameInfo.FileName[:], p16)
	renameInfo.FileNameLength = renameInfoEx.FileNameLength

	err = NtSetInformationFile(
		h,
		&IO_STATUS_BLOCK{},
		unsafe.Pointer(&renameInfo),
		uint32(unsafe.Sizeof(FILE_RENAME_INFORMATION{})),
		FileRenameInformation,
	)
	if st, ok := err.(NTStatus); ok {
		return st.Errno()
	}
	return err
}

func Linkat(olddirfd syscall.Handle, oldpath string, newdirfd syscall.Handle, newpath string) error {
	objAttrs := &OBJECT_ATTRIBUTES{}
	if err := objAttrs.init(olddirfd, oldpath); err != nil {
		return err
	}
	var h syscall.Handle
	err := NtOpenFile(
		&h,
		SYNCHRONIZE|FILE_WRITE_ATTRIBUTES,
		objAttrs,
		&IO_STATUS_BLOCK{},
		FILE_SHARE_DELETE|FILE_SHARE_READ|FILE_SHARE_WRITE,
		FILE_OPEN_REPARSE_POINT|FILE_OPEN_FOR_BACKUP_INTENT|FILE_SYNCHRONOUS_IO_NONALERT,
	)
	if err != nil {
		return ntCreateFileError(err, 0)
	}
	defer syscall.CloseHandle(h)

	linkInfo := FILE_LINK_INFORMATION{
		RootDirectory: newdirfd,
	}
	p16, err := syscall.UTF16FromString(newpath)
	if err != nil {
		return err
	}
	if len(p16) > len(linkInfo.FileName) {
		return syscall.EINVAL
	}
	copy(linkInfo.FileName[:], p16)
	linkInfo.FileNameLength = uint32((len(p16) - 1) * 2)

	const (
		FileLinkInformation = 11
	)
	err = NtSetInformationFile(
		h,
		&IO_STATUS_BLOCK{},
		unsafe.Pointer(&linkInfo),
		uint32(unsafe.Sizeof(FILE_LINK_INFORMATION{})),
		FileLinkInformation,
	)
	if st, ok := err.(NTStatus); ok {
		return st.Errno()
	}
	return err
}

// SymlinkatFlags configure Symlinkat.
//
// Symbolic links have two properties: They may be directory or file links,
// and they may be absolute or relative.
//
// The Windows API defines flags describing these properties
// (SYMBOLIC_LINK_FLAG_DIRECTORY and SYMLINK_FLAG_RELATIVE),
// but the flags are passed to different system calls and
// do not have distinct values, so we define our own enumeration
// that permits expressing both.
type SymlinkatFlags uint

const (
	SYMLINKAT_DIRECTORY = SymlinkatFlags(1 << iota)
	SYMLINKAT_RELATIVE
)

func Symlinkat(oldname string, newdirfd syscall.Handle, newname string, flags SymlinkatFlags) error {
	// Temporarily acquire symlink-creating privileges if possible.
	// This is the behavior of CreateSymbolicLinkW.
	//
	// (When passed the SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE flag,
	// CreateSymbolicLinkW ignores errors in acquiring privileges, as we do here.)
	return withPrivilege("SeCreateSymbolicLinkPrivilege", func() error {
		return symlinkat(oldname, newdirfd, newname, flags)
	})
}

func symlinkat(oldname string, newdirfd syscall.Handle, newname string, flags SymlinkatFlags) error {
	oldnameu16, err := syscall.UTF16FromString(oldname)
	if err != nil {
		return err
	}
	oldnameu16 = oldnameu16[:len(oldnameu16)-1] // trim off terminal NUL

	var options uint32
	if flags&SYMLINKAT_DIRECTORY != 0 {
		options |= FILE_DIRECTORY_FILE
	} else {
		options |= FILE_NON_DIRECTORY_FILE
	}

	objAttrs := &OBJECT_ATTRIBUTES{}
	if err := objAttrs.init(newdirfd, newname); err != nil {
		return err
	}
	var h syscall.Handle
	err = NtCreateFile(
		&h,
		SYNCHRONIZE|FILE_WRITE_ATTRIBUTES|DELETE,
		objAttrs,
		&IO_STATUS_BLOCK{},
		nil,
		syscall.FILE_ATTRIBUTE_NORMAL,
		0,
		FILE_CREATE,
		FILE_OPEN_REPARSE_POINT|FILE_OPEN_FOR_BACKUP_INTENT|FILE_SYNCHRONOUS_IO_NONALERT|options,
		nil,
		0,
	)
	if err != nil {
		return ntCreateFileError(err, 0)
	}
	defer syscall.CloseHandle(h)

	// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_reparse_data_buffer
	type reparseDataBufferT struct {
		_ structs.HostLayout

		ReparseTag        uint32
		ReparseDataLength uint16
		Reserved          uint16

		SubstituteNameOffset uint16
		SubstituteNameLength uint16
		PrintNameOffset      uint16
		PrintNameLength      uint16
		Flags                uint32
	}

	const (
		headerSize = uint16(unsafe.Offsetof(reparseDataBufferT{}.SubstituteNameOffset))
		bufferSize = uint16(unsafe.Sizeof(reparseDataBufferT{}))
	)

	// Data buffer containing a SymbolicLinkReparseBuffer followed by the link target.
	rdbbuf := make([]byte, bufferSize+uint16(2*len(oldnameu16)))

	rdb := (*reparseDataBufferT)(unsafe.Pointer(&rdbbuf[0]))
	rdb.ReparseTag = syscall.IO_REPARSE_TAG_SYMLINK
	rdb.ReparseDataLength = uint16(len(rdbbuf)) - uint16(headerSize)
	rdb.SubstituteNameOffset = 0
	rdb.SubstituteNameLength = uint16(2 * len(oldnameu16))
	rdb.PrintNameOffset = 0
	rdb.PrintNameLength = rdb.SubstituteNameLength
	if flags&SYMLINKAT_RELATIVE != 0 {
		rdb.Flags = SYMLINK_FLAG_RELATIVE
	}

	namebuf := rdbbuf[bufferSize:]
	copy(namebuf, unsafe.String((*byte)(unsafe.Pointer(&oldnameu16[0])), 2*len(oldnameu16)))

	err = syscall.DeviceIoControl(
		h,
		FSCTL_SET_REPARSE_POINT,
		&rdbbuf[0],
		uint32(len(rdbbuf)),
		nil,
		0,
		nil,
		nil)
	if err != nil {
		// Creating the symlink has failed, so try to remove the file.
		const FileDispositionInformation = 13
		NtSetInformationFile(
			h,
			&IO_STATUS_BLOCK{},
			unsafe.Pointer(&FILE_DISPOSITION_INFORMATION{
				DeleteFile: true,
			}),
			uint32(unsafe.Sizeof(FILE_DISPOSITION_INFORMATION{})),
			FileDispositionInformation,
		)
		return err
	}

	return nil
}

// withPrivilege temporariliy acquires the named privilege and runs f.
// If the privilege cannot be acquired it runs f anyway,
// which should fail with an appropriate error.
func withPrivilege(privilege string, f func() error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	err := ImpersonateSelf(SecurityImpersonation)
	if err != nil {
		return f()
	}
	defer RevertToSelf()

	curThread, err := GetCurrentThread()
	if err != nil {
		return f()
	}
	var token syscall.Token
	err = OpenThreadToken(curThread, syscall.TOKEN_QUERY|TOKEN_ADJUST_PRIVILEGES, false, &token)
	if err != nil {
		return f()
	}
	defer syscall.CloseHandle(syscall.Handle(token))

	privStr, err := syscall.UTF16PtrFromString(privilege)
	if err != nil {
		return f()
	}
	var tokenPriv TOKEN_PRIVILEGES
	err = LookupPrivilegeValue(nil, privStr, &tokenPriv.Privileges[0].Luid)
	if err != nil {
		return f()
	}

	tokenPriv.PrivilegeCount = 1
	tokenPriv.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
	err = AdjustTokenPrivileges(token, false, &tokenPriv, 0, nil, nil)
	if err != nil {
		return f()
	}

	return f()
}
