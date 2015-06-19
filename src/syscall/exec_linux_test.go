// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package syscall_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"testing"
)

func whoamiCmd(t *testing.T, uid, gid int, setgroups bool) *exec.Cmd {
	if _, err := os.Stat("/proc/self/ns/user"); err != nil {
		if os.IsNotExist(err) {
			t.Skip("kernel doesn't support user namespaces")
		}
		t.Fatalf("Failed to stat /proc/self/ns/user: %v", err)
	}
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
		// On some systems, there is a sysctl setting.
		if os.IsPermission(err) && os.Getuid() != 0 {
			data, errRead := ioutil.ReadFile("/proc/sys/kernel/unprivileged_userns_clone")
			if errRead == nil && data[0] == '0' {
				t.Skip("kernel prohibits user namespace in unprivileged process")
			}
		}

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

// kernelVersion returns the major and minor versions of the Linux
// kernel version.  It calls t.Skip if it can't figure it out.
func kernelVersion(t *testing.T) (int, int) {
	bytes, err := ioutil.ReadFile("/proc/version")
	if err != nil {
		t.Skipf("can't get kernel version: %v", err)
	}
	matches := regexp.MustCompile("([0-9]+).([0-9]+)").FindSubmatch(bytes)
	if len(matches) < 3 {
		t.Skipf("can't get kernel version from %s", bytes)
	}
	major, _ := strconv.Atoi(string(matches[1]))
	minor, _ := strconv.Atoi(string(matches[2]))
	return major, minor
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
