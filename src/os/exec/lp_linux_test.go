// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"internal/syscall/unix"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func TestFindExecutableVsNoexec(t *testing.T) {
	// This test case relies on faccessat2(2) syscall, which appeared in Linux v5.8.
	if major, minor := unix.KernelVersion(); major < 5 || (major == 5 && minor < 8) {
		t.Skip("requires Linux kernel v5.8 with faccessat2(2) syscall")
	}

	tmp := t.TempDir()

	// Create a tmpfs mount.
	err := syscall.Mount("tmpfs", tmp, "tmpfs", 0, "")
	if err != nil {
		// Usually this means lack of CAP_SYS_ADMIN, but there might be
		// other reasons, expecially in restricted test environments.
		t.Skipf("requires ability to mount tmpfs (%v)", err)
	}
	t.Cleanup(func() {
		if err := syscall.Unmount(tmp, 0); err != nil {
			t.Error(err)
		}
	})

	// Create an executable.
	path := filepath.Join(tmp, "program")
	err = os.WriteFile(path, []byte("#!/bin/sh\necho 123\n"), 0o755)
	if err != nil {
		t.Fatal(err)
	}

	// Check that it works as expected.
	err = findExecutable(path)
	if err != nil {
		t.Fatalf("findExecutable: got %v, want nil", err)
	}

	if err := Command(path).Run(); err != nil {
		t.Fatalf("exec: got %v, want nil", err)
	}

	// Remount with noexec flag.
	err = syscall.Mount("", tmp, "", syscall.MS_REMOUNT|syscall.MS_NOEXEC, "")
	if err != nil {
		t.Fatalf("remount %s with noexec failed: %v", tmp, err)
	}

	if err := Command(path).Run(); err == nil {
		t.Fatal("exec on noexec filesystem: got nil, want error")
	}

	err = findExecutable(path)
	if err == nil {
		t.Fatalf("findExecutable: got nil, want error")
	}
}
