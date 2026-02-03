// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows_test

import (
	"internal/syscall/windows"
	"internal/syscall/windows/sysdll"

	"os"
	"path/filepath"
	"syscall"
	"testing"
	"unsafe"
)

var (
	modadvapi32               = syscall.NewLazyDLL(sysdll.Add("advapi32.dll"))
	procSetEntriesInAclW      = modadvapi32.NewProc("SetEntriesInAclW")
	procSetNamedSecurityInfoW = modadvapi32.NewProc("SetNamedSecurityInfoW")
)

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379636.aspx
type trustee struct {
	MultipleTrustee          *trustee
	MultipleTrusteeOperation int32
	TrusteeForm              int32
	TrusteeType              int32
	Name                     *uint16
}

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa446627.aspx
type explicitAccess struct {
	AccessPermissions uint32
	AccessMode        int32
	Inheritance       uint32
	Trustee           trustee
}

const (
	// https://msdn.microsoft.com/en-us/library/windows/desktop/aa374899.aspx
	GRANT_ACCESS = 1

	// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379638.aspx
	TRUSTEE_IS_SID = 0

	// https://msdn.microsoft.com/en-us/library/windows/desktop/aa446627.aspx
	SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3

	// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379593.aspx
	SE_FILE_OBJECT = 1

	// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379573.aspx
	DACL_SECURITY_INFORMATION           = 0x00004
	PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
)

func TestOpen(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	file := filepath.Join(dir, "a")
	f, err := os.Create(file)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	tests := []struct {
		path string
		flag int
		err  error
	}{
		{dir, syscall.O_RDONLY, nil},
		{dir, syscall.O_CREAT, nil},
		{dir, syscall.O_RDONLY | syscall.O_CREAT, nil},
		{file, syscall.O_APPEND | syscall.O_WRONLY | os.O_CREATE, nil},
		{file, syscall.O_APPEND | syscall.O_WRONLY | os.O_CREATE | os.O_TRUNC, nil},
		{dir, syscall.O_RDONLY | syscall.O_TRUNC, syscall.ERROR_ACCESS_DENIED},
		{dir, syscall.O_WRONLY | syscall.O_RDWR, nil}, // TODO: syscall.Open returns EISDIR here, we should reconcile this
		{dir, syscall.O_WRONLY, syscall.EISDIR},
		{dir, syscall.O_RDWR, syscall.EISDIR},
	}
	for i, tt := range tests {
		dir := filepath.Dir(tt.path)
		dirfd, err := syscall.Open(dir, syscall.O_RDONLY, 0)
		if err != nil {
			t.Error(err)
			continue
		}
		base := filepath.Base(tt.path)
		h, err := windows.Openat(dirfd, base, uint64(tt.flag), 0o660)
		syscall.CloseHandle(dirfd)
		if err == nil {
			syscall.CloseHandle(h)
		}
		if err != tt.err {
			t.Errorf("%d: Open got %q, want %q", i, err, tt.err)
		}
	}
}

func TestDeleteAt(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		modifier func(t *testing.T, path string)
	}{
		{"DeleteAt removes normal file", func(t *testing.T, name string) {}},
		{"DeleteAt removes file with no read permission", makeFileNotReadable},
		{"DeleteAt removes readonly file", makeFileReadonly},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {

			dir := t.TempDir()
			file := filepath.Join(dir, "a")
			f, err := os.Create(file)
			if err != nil {
				t.Fatal(err)
			}
			f.Close()

			// Remove all permissions from the file.
			// Do not use os.Chmod it sets only readonly attribute on Windows.
			tc.modifier(t, file)

			// delete file using DeleteAt
			dirfd, err := syscall.Open(dir, syscall.O_RDONLY, 0)
			if err != nil {
				t.Fatal(err)
			}
			base := filepath.Base(file)
			err = windows.Deleteat(dirfd, base, 0)
			syscall.CloseHandle(dirfd)
			if err != nil {
				t.Fatalf("Deleteat failed: %v", err)
			}

			// Verify the file has been deleted.
			if _, err := os.Stat(file); !os.IsNotExist(err) {
				t.Fatalf("file still exists after DeleteAt")
			}
		})
	}
}

func makeFileReadonly(t *testing.T, name string) {
	if err := os.Chmod(name, 0); err != nil {
		t.Fatal(err)
	}
}

func makeFileNotReadable(t *testing.T, name string) {
	creatorOwnerSID, err := syscall.StringToSid("S-1-3-0")
	if err != nil {
		t.Fatal(err)
	}
	creatorGroupSID, err := syscall.StringToSid("S-1-3-1")
	if err != nil {
		t.Fatal(err)
	}
	everyoneSID, err := syscall.StringToSid("S-1-1-0")
	if err != nil {
		t.Fatal(err)
	}

	entryForSid := func(sid *syscall.SID) explicitAccess {
		return explicitAccess{
			AccessPermissions: 0,
			AccessMode:        GRANT_ACCESS,
			Inheritance:       SUB_CONTAINERS_AND_OBJECTS_INHERIT,
			Trustee: trustee{
				TrusteeForm: TRUSTEE_IS_SID,
				Name:        (*uint16)(unsafe.Pointer(sid)),
			},
		}
	}

	entries := []explicitAccess{
		entryForSid(creatorOwnerSID),
		entryForSid(creatorGroupSID),
		entryForSid(everyoneSID),
	}

	var oldAcl, newAcl syscall.Handle
	if err := setEntriesInAcl(
		entries,
		oldAcl,
		&newAcl,
	); err != nil {
		t.Fatal(err)
	}

	defer syscall.LocalFree((syscall.Handle)(unsafe.Pointer(newAcl)))
	if err := setNamedSecurityInfo(
		name,
		SE_FILE_OBJECT,
		DACL_SECURITY_INFORMATION|PROTECTED_DACL_SECURITY_INFORMATION,
		nil,
		nil,
		newAcl,
		0,
	); err != nil {
		t.Fatal(err)
	}
}

func setEntriesInAcl(entries []explicitAccess, oldAcl syscall.Handle, newAcl *syscall.Handle) error {
	ret, _, _ := procSetEntriesInAclW.Call(
		uintptr(len(entries)),
		uintptr(unsafe.Pointer(&entries[0])),
		uintptr(oldAcl),
		uintptr(unsafe.Pointer(newAcl)),
	)
	if ret != 0 {
		return syscall.Errno(ret)
	}
	return nil
}

func setNamedSecurityInfo(objectName string, objectType int32, secInfo uint32, owner, group *syscall.SID, dacl, sacl syscall.Handle) error {
	ret, _, _ := procSetNamedSecurityInfoW.Call(
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(objectName))),
		uintptr(objectType),
		uintptr(secInfo),
		uintptr(unsafe.Pointer(owner)),
		uintptr(unsafe.Pointer(group)),
		uintptr(dacl),
		uintptr(sacl),
	)
	if ret != 0 {
		return syscall.Errno(ret)
	}
	return nil
}
