// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"internal/syscall/windows"
	"io/fs"
	"os"
	"unsafe"
)

// move moves a file from src to dst, setting the security information
// on the destination file to inherit the permissions from the
// destination parent directory.
func (sh *Shell) move(src, dst string, perm fs.FileMode) (err error) {
	if err := os.Rename(src, dst); err != nil {
		return err
	}
	defer func() {
		if err != nil {
			os.Remove(dst) // clean up if we failed to set the mode or security info
		}
	}()
	if err := os.Chmod(dst, perm); err != nil {
		return err
	}
	// We need to respect the ACL permissions of the destination parent folder.
	// https://go.dev/issue/22343.
	var acl windows.ACL
	if err := windows.InitializeAcl(&acl, uint32(unsafe.Sizeof(acl)), windows.ACL_REVISION); err != nil {
		return err
	}
	secInfo := windows.DACL_SECURITY_INFORMATION | windows.UNPROTECTED_DACL_SECURITY_INFORMATION
	return windows.SetNamedSecurityInfo(dst, windows.SE_FILE_OBJECT, secInfo, nil, nil, &acl, nil)
}
