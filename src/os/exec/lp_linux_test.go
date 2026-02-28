// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec_test

import (
	"errors"
	"internal/syscall/unix"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
)

func TestFindExecutableVsNoexec(t *testing.T) {
	t.Parallel()

	// This test case relies on faccessat2(2) syscall, which appeared in Linux v5.8.
	if !unix.KernelVersionGE(5, 8) {
		t.Skip("requires Linux kernel v5.8 with faccessat2(2) syscall")
	}

	tmp := t.TempDir()

	// Create a tmpfs mount.
	err := syscall.Mount("tmpfs", tmp, "tmpfs", 0, "")
	if testenv.SyscallIsNotSupported(err) {
		// Usually this means lack of CAP_SYS_ADMIN, but there might be
		// other reasons, especially in restricted test environments.
		t.Skipf("requires ability to mount tmpfs (%v)", err)
	} else if err != nil {
		t.Fatalf("mount %s failed: %v", tmp, err)
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
	_, err = exec.LookPath(path)
	if err != nil {
		t.Fatalf("LookPath: got %v, want nil", err)
	}

	for {
		err = exec.Command(path).Run()
		if err == nil {
			break
		}
		if errors.Is(err, syscall.ETXTBSY) {
			// A fork+exec in another process may be holding open the FD that we used
			// to write the executable (see https://go.dev/issue/22315).
			// Since the descriptor should have CLOEXEC set, the problem should resolve
			// as soon as the forked child reaches its exec call.
			// Keep retrying until that happens.
		} else {
			t.Fatalf("exec: got %v, want nil", err)
		}
	}

	// Remount with noexec flag.
	err = syscall.Mount("", tmp, "", syscall.MS_REMOUNT|syscall.MS_NOEXEC, "")
	if testenv.SyscallIsNotSupported(err) {
		t.Skipf("requires ability to re-mount tmpfs (%v)", err)
	} else if err != nil {
		t.Fatalf("remount %s with noexec failed: %v", tmp, err)
	}

	if err := exec.Command(path).Run(); err == nil {
		t.Fatal("exec on noexec filesystem: got nil, want error")
	}

	_, err = exec.LookPath(path)
	if err == nil {
		t.Fatalf("LookPath: got nil, want error")
	}
}
