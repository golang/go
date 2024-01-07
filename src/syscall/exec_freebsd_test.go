// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd

package syscall_test

import (
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"unsafe"
)

const (
	flagJailCreate = uintptr(0x1)
)

func prepareJail(t *testing.T) (int, string) {
	t.Helper()

	root := t.TempDir()
	paramPath := []byte("path\x00")
	conf := make([]syscall.Iovec, 4)
	conf[0].Base = &paramPath[0]
	conf[0].SetLen(len(paramPath))
	p, err := syscall.BytePtrFromString(root)
	if err != nil {
		t.Fatal(err)
	}
	conf[1].Base = p
	conf[1].SetLen(len(root) + 1)

	paramPersist := []byte("persist\x00")
	conf[2].Base = &paramPersist[0]
	conf[2].SetLen(len(paramPersist))
	conf[3].Base = nil
	conf[3].SetLen(0)

	id, _, err1 := syscall.Syscall(syscall.SYS_JAIL_SET,
		uintptr(unsafe.Pointer(&conf[0])), uintptr(len(conf)), flagJailCreate)
	if err1 != 0 {
		t.Fatalf("jail_set: %v", err1)
	}
	t.Cleanup(func() {
		_, _, err1 := syscall.Syscall(syscall.SYS_JAIL_REMOVE, id, 0, 0)
		if err1 != 0 {
			t.Errorf("failed to cleanup jail: %v", err)
		}
	})

	return int(id), root
}

func TestJailAttach(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		jailed, err := syscall.SysctlUint32("security.jail.jailed")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		if jailed != 1 {
			t.Fatalf("jailed = %d, want 1", jailed)
		}
		return
	}

	testenv.MustHaveGoBuild(t)
	// Make sure we are running as root, so we have permissions to create
	// and remove jails.
	if os.Getuid() != 0 {
		t.Skip("kernel prohibits jail system calls in unprivileged process")
	}

	jid, root := prepareJail(t)

	// Since jail attach does an implicit chroot to the jail's path,
	// we need the binary there, and it must be statically linked.
	x := filepath.Join(root, "syscall.test")
	cmd := exec.Command(testenv.GoToolPath(t), "test", "-c", "-o", x, "syscall")
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	if o, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Build of syscall in jail root failed, output %v, err %v", o, err)
	}

	cmd = exec.Command("/syscall.test", "-test.run=TestJailAttach", "/")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	cmd.SysProcAttr = &syscall.SysProcAttr{Jail: jid}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Cmd failed with err %v, output: %s", err, out)
	}
}
