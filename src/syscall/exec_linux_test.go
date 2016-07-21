// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package syscall_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"
)

// Check if we are in a chroot by checking if the inode of / is
// different from 2 (there is no better test available to non-root on
// linux).
func isChrooted(t *testing.T) bool {
	root, err := os.Stat("/")
	if err != nil {
		t.Fatalf("cannot stat /: %v", err)
	}
	return root.Sys().(*syscall.Stat_t).Ino != 2
}

func checkUserNS(t *testing.T) {
	if _, err := os.Stat("/proc/self/ns/user"); err != nil {
		if os.IsNotExist(err) {
			t.Skip("kernel doesn't support user namespaces")
		}
		if os.IsPermission(err) {
			t.Skip("unable to test user namespaces due to permissions")
		}
		t.Fatalf("Failed to stat /proc/self/ns/user: %v", err)
	}
	if isChrooted(t) {
		// create_user_ns in the kernel (see
		// https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/kernel/user_namespace.c)
		// forbids the creation of user namespaces when chrooted.
		t.Skip("cannot create user namespaces when chrooted")
	}
	// On some systems, there is a sysctl setting.
	if os.Getuid() != 0 {
		data, errRead := ioutil.ReadFile("/proc/sys/kernel/unprivileged_userns_clone")
		if errRead == nil && data[0] == '0' {
			t.Skip("kernel prohibits user namespace in unprivileged process")
		}
	}
	// When running under the Go continuous build, skip tests for
	// now when under Kubernetes. (where things are root but not quite)
	// Both of these are our own environment variables.
	// See Issue 12815.
	if os.Getenv("GO_BUILDER_NAME") != "" && os.Getenv("IN_KUBERNETES") == "1" {
		t.Skip("skipping test on Kubernetes-based builders; see Issue 12815")
	}
}

func whoamiCmd(t *testing.T, uid, gid int, setgroups bool) *exec.Cmd {
	checkUserNS(t)
	cmd := exec.Command("whoami")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWUSER,
		UidMappings: []syscall.SysProcIDMap{
			{ContainerID: 0, HostID: uid, Size: 1},
		},
		GidMappings: []syscall.SysProcIDMap{
			{ContainerID: 0, HostID: gid, Size: 1},
		},
		GidMappingsEnableSetgroups: setgroups,
	}
	return cmd
}

func testNEWUSERRemap(t *testing.T, uid, gid int, setgroups bool) {
	cmd := whoamiCmd(t, uid, gid, setgroups)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Cmd failed with err %v, output: %s", err, out)
	}
	sout := strings.TrimSpace(string(out))
	want := "root"
	if sout != want {
		t.Fatalf("whoami = %q; want %q", out, want)
	}
}

func TestCloneNEWUSERAndRemapRootDisableSetgroups(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("skipping root only test")
	}
	testNEWUSERRemap(t, 0, 0, false)
}

func TestCloneNEWUSERAndRemapRootEnableSetgroups(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("skipping root only test")
	}
	testNEWUSERRemap(t, 0, 0, false)
}

func TestCloneNEWUSERAndRemapNoRootDisableSetgroups(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("skipping unprivileged user only test")
	}
	testNEWUSERRemap(t, os.Getuid(), os.Getgid(), false)
}

func TestCloneNEWUSERAndRemapNoRootSetgroupsEnableSetgroups(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("skipping unprivileged user only test")
	}
	cmd := whoamiCmd(t, os.Getuid(), os.Getgid(), true)
	err := cmd.Run()
	if err == nil {
		t.Skip("probably old kernel without security fix")
	}
	if !os.IsPermission(err) {
		t.Fatalf("Unprivileged gid_map rewriting with GidMappingsEnableSetgroups must fail")
	}
}

func TestEmptyCredGroupsDisableSetgroups(t *testing.T) {
	cmd := whoamiCmd(t, os.Getuid(), os.Getgid(), false)
	cmd.SysProcAttr.Credential = &syscall.Credential{}
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
}

func TestUnshare(t *testing.T) {
	// Make sure we are running as root so we have permissions to use unshare
	// and create a network namespace.
	if os.Getuid() != 0 {
		t.Skip("kernel prohibits unshare in unprivileged process, unless using user namespace")
	}

	// When running under the Go continuous build, skip tests for
	// now when under Kubernetes. (where things are root but not quite)
	// Both of these are our own environment variables.
	// See Issue 12815.
	if os.Getenv("GO_BUILDER_NAME") != "" && os.Getenv("IN_KUBERNETES") == "1" {
		t.Skip("skipping test on Kubernetes-based builders; see Issue 12815")
	}

	path := "/proc/net/dev"
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Skip("kernel doesn't support proc filesystem")
		}
		if os.IsPermission(err) {
			t.Skip("unable to test proc filesystem due to permissions")
		}
		t.Fatal(err)
	}
	if _, err := os.Stat("/proc/self/ns/net"); err != nil {
		if os.IsNotExist(err) {
			t.Skip("kernel doesn't support net namespace")
		}
		t.Fatal(err)
	}

	cmd := exec.Command("cat", path)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Unshareflags: syscall.CLONE_NEWNET,
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Cmd failed with err %v, output: %s", err, out)
	}

	// Check there is only the local network interface
	sout := strings.TrimSpace(string(out))
	if !strings.Contains(sout, "lo:") {
		t.Fatalf("Expected lo network interface to exist, got %s", sout)
	}

	lines := strings.Split(sout, "\n")
	if len(lines) != 3 {
		t.Fatalf("Expected 3 lines of output, got %d", len(lines))
	}
}

func TestGroupCleanup(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("we need root for credential")
	}
	cmd := exec.Command("id")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Credential: &syscall.Credential{
			Uid: 0,
			Gid: 0,
		},
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Cmd failed with err %v, output: %s", err, out)
	}
	strOut := strings.TrimSpace(string(out))
	expected := "uid=0(root) gid=0(root) groups=0(root)"
	// Just check prefix because some distros reportedly output a
	// context parameter; see https://golang.org/issue/16224.
	if !strings.HasPrefix(strOut, expected) {
		t.Errorf("id command output: %q, expected prefix: %q", strOut, expected)
	}
}

func TestGroupCleanupUserNamespace(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("we need root for credential")
	}
	checkUserNS(t)
	cmd := exec.Command("id")
	uid, gid := os.Getuid(), os.Getgid()
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWUSER,
		Credential: &syscall.Credential{
			Uid: uint32(uid),
			Gid: uint32(gid),
		},
		UidMappings: []syscall.SysProcIDMap{
			{ContainerID: 0, HostID: uid, Size: 1},
		},
		GidMappings: []syscall.SysProcIDMap{
			{ContainerID: 0, HostID: gid, Size: 1},
		},
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Cmd failed with err %v, output: %s", err, out)
	}
	strOut := strings.TrimSpace(string(out))

	// Strings we've seen in the wild.
	expected := []string{
		"uid=0(root) gid=0(root) groups=0(root)",
		"uid=0(root) gid=0(root) groups=0(root),65534(nobody)",
		"uid=0(root) gid=0(root) groups=0(root),65534(nogroup)",
		"uid=0(root) gid=0(root) groups=0(root),65534",
	}
	for _, e := range expected {
		if strOut == e {
			return
		}
	}
	t.Errorf("id command output: %q, expected one of %q", strOut, expected)
}
