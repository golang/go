// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package toolchain

import (
	"internal/syscall/unix"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"

	"cmd/go/internal/gover"
)

// pathDirs returns the directories in the system search path.
func pathDirs() []string {
	return filepath.SplitList(os.Getenv("PATH"))
}

// pathVersion returns the Go version implemented by the file
// described by de and info in directory dir.
// The analysis only uses the name itself; it does not run the program.
func pathVersion(dir string, de fs.DirEntry, info fs.FileInfo) (string, bool) {
	v := gover.FromToolchain(de.Name())
	if v == "" {
		return "", false
	}

	// Mimicking exec.findExecutable here.
	// ENOSYS means Eaccess is not available or not implemented.
	// EPERM can be returned by Linux containers employing seccomp.
	// In both cases, fall back to checking the permission bits.
	err := unix.Eaccess(filepath.Join(dir, de.Name()), unix.X_OK)
	if (err == syscall.ENOSYS || err == syscall.EPERM) && info.Mode()&0111 != 0 {
		err = nil
	}
	if err != nil {
		return "", false
	}

	return v, true
}
