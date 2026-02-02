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

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379638.aspx
const (
	TRUSTEE_IS_SID = iota
)

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa446627.aspx
const (
	SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
)


// https://msdn.microsoft.com/en-us/library/windows/desktop/aa374899.aspx
const (
	NOT_USED_ACCESS = iota
	GRANT_ACCESS
)


var (
	modadvapi32 = syscall.NewLazyDLL(sysdll.Add("advapi32.dll"))
	procSetEntriesInAclW = modadvapi32.NewProc("SetEntriesInAclW")
	procSetNamedSecurityInfoW = modadvapi32.NewProc("SetNamedSecurityInfoW")
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


func TestDeleteAt_InaccessibleFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	file := filepath.Join(dir, "a")
	f, err := os.Create(file)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	// Remove all permissions from the file.
	// Do not use os.Chmod it sets only readonly attribute on Windows.
	if err := makeFileInaccessible(t, file); err != nil {
		t.Fatalf("makeFileInaccessible failed: %v", err)
	}

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
}

func makeFileInaccessible(t *testing.T, path string) error {
	creatorOwnerSID, err := syscall.StringToSid("S-1-3-0")
	if err != nil {
		return err
	}
	creatorGroupSID, err := syscall.StringToSid("S-1-3-1")
	if err != nil {
		return err
	}
	everyoneSID, err := syscall.StringToSid("S-1-1-0")
	if err != nil {
		return err
	}

	var permissions uint32 = 0

	// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379636.aspx
	aclEntries := []explicitAccess{
		{
			AccessPermissions: permissions,
			AccessMode:        GRANT_ACCESS,
			Inheritance:       SUB_CONTAINERS_AND_OBJECTS_INHERIT,
			Trustee: trustee{
				TrusteeForm: TRUSTEE_IS_SID, 
				Name:        (*uint16)(unsafe.Pointer(creatorOwnerSID)),
			},
		},
		{
			AccessPermissions: permissions,
			AccessMode:        GRANT_ACCESS,
			Inheritance:       SUB_CONTAINERS_AND_OBJECTS_INHERIT,
			Trustee: trustee{
				TrusteeForm: TRUSTEE_IS_SID,
				Name:        (*uint16)(unsafe.Pointer(creatorGroupSID)),
			},
		},
		{
			AccessPermissions: permissions,
			AccessMode:        GRANT_ACCESS,
			Inheritance:       SUB_CONTAINERS_AND_OBJECTS_INHERIT,
			Trustee: trustee{
				TrusteeForm: TRUSTEE_IS_SID, 
				Name:        (*uint16)(unsafe.Pointer(everyoneSID)),
			},
		},
	}

	var oldAcl, newAcl syscall.Handle
	ret, _, _ :=  syscall.SyscallN(
		procSetEntriesInAclW.Addr(),
		uintptr(len(aclEntries)),
		uintptr(unsafe.Pointer(&aclEntries[0])),
		uintptr(oldAcl),
		uintptr(unsafe.Pointer(newAcl)),
	)
	
	if ret != 0 {
		return syscall.Errno(ret)
	}

	err = setNamedSecurityInfo(
		path,
		1, // SE_FILE_OBJECT
		0x00004|0x20000000, // DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION
		nil, 
		nil, 
		newAcl, 
		0,
	) 
	if err != nil {
		return err
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
		uintptr(unsafe.Pointer(dacl)),
		uintptr(sacl),
	)
	if ret != 0 {
		return syscall.Errno(ret)
	}
	return nil
}


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

