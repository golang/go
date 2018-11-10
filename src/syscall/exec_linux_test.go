// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package syscall_test

import (
	"flag"
	"fmt"
	"internal/testenv"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"
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
	// On Centos 7 make sure they set the kernel parameter user_namespace=1
	// See issue 16283 and 20796.
	if _, err := os.Stat("/sys/module/user_namespace/parameters/enable"); err == nil {
		buf, _ := ioutil.ReadFile("/sys/module/user_namespace/parameters/enabled")
		if !strings.HasPrefix(string(buf), "Y") {
			t.Skip("kernel doesn't support user namespaces")
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

	orig, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	origLines := strings.Split(strings.TrimSpace(string(orig)), "\n")

	cmd := exec.Command("cat", path)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Unshareflags: syscall.CLONE_NEWNET,
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		if strings.Contains(err.Error(), "operation not permitted") {
			// Issue 17206: despite all the checks above,
			// this still reportedly fails for some users.
			// (older kernels?). Just skip.
			t.Skip("skipping due to permission error")
		}
		t.Fatalf("Cmd failed with err %v, output: %s", err, out)
	}

	// Check there is only the local network interface
	sout := strings.TrimSpace(string(out))
	if !strings.Contains(sout, "lo:") {
		t.Fatalf("Expected lo network interface to exist, got %s", sout)
	}

	lines := strings.Split(sout, "\n")
	if len(lines) >= len(origLines) {
		t.Fatalf("Got %d lines of output, want <%d", len(lines), len(origLines))
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
	expected := "uid=0(root) gid=0(root)"
	// Just check prefix because some distros reportedly output a
	// context parameter; see https://golang.org/issue/16224.
	// Alpine does not output groups; see https://golang.org/issue/19938.
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
		"uid=0(root) gid=0(root) groups=0(root),65534(nobody),65534(nobody),65534(nobody),65534(nobody),65534(nobody),65534(nobody),65534(nobody),65534(nobody),65534(nobody),65534(nobody)", // Alpine; see https://golang.org/issue/19938
	}
	for _, e := range expected {
		if strOut == e {
			return
		}
	}
	t.Errorf("id command output: %q, expected one of %q", strOut, expected)
}

// TestUnshareHelperProcess isn't a real test. It's used as a helper process
// for TestUnshareMountNameSpace.
func TestUnshareMountNameSpaceHelper(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)
	if err := syscall.Mount("none", flag.Args()[0], "proc", 0, ""); err != nil {
		fmt.Fprintf(os.Stderr, "unshare: mount %v failed: %v", os.Args, err)
		os.Exit(2)
	}
}

// Test for Issue 38471: unshare fails because systemd has forced / to be shared
func TestUnshareMountNameSpace(t *testing.T) {
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

	d, err := ioutil.TempDir("", "unshare")
	if err != nil {
		t.Fatalf("tempdir: %v", err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestUnshareMountNameSpaceHelper", d)
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	cmd.SysProcAttr = &syscall.SysProcAttr{Unshareflags: syscall.CLONE_NEWNS}

	o, err := cmd.CombinedOutput()
	if err != nil {
		if strings.Contains(err.Error(), ": permission denied") {
			t.Skipf("Skipping test (golang.org/issue/19698); unshare failed due to permissions: %s, %v", o, err)
		}
		t.Fatalf("unshare failed: %s, %v", o, err)
	}

	// How do we tell if the namespace was really unshared? It turns out
	// to be simple: just try to remove the directory. If it's still mounted
	// on the rm will fail with EBUSY. Then we have some cleanup to do:
	// we must unmount it, then try to remove it again.

	if err := os.Remove(d); err != nil {
		t.Errorf("rmdir failed on %v: %v", d, err)
		if err := syscall.Unmount(d, syscall.MNT_FORCE); err != nil {
			t.Errorf("Can't unmount %v: %v", d, err)
		}
		if err := os.Remove(d); err != nil {
			t.Errorf("rmdir after unmount failed on %v: %v", d, err)
		}
	}
}

// Test for Issue 20103: unshare fails when chroot is used
func TestUnshareMountNameSpaceChroot(t *testing.T) {
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

	d, err := ioutil.TempDir("", "unshare")
	if err != nil {
		t.Fatalf("tempdir: %v", err)
	}

	// Since we are doing a chroot, we need the binary there,
	// and it must be statically linked.
	x := filepath.Join(d, "syscall.test")
	cmd := exec.Command(testenv.GoToolPath(t), "test", "-c", "-o", x, "syscall")
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	if o, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Build of syscall in chroot failed, output %v, err %v", o, err)
	}

	cmd = exec.Command("/syscall.test", "-test.run=TestUnshareMountNameSpaceHelper", "/")
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	cmd.SysProcAttr = &syscall.SysProcAttr{Chroot: d, Unshareflags: syscall.CLONE_NEWNS}

	o, err := cmd.CombinedOutput()
	if err != nil {
		if strings.Contains(err.Error(), ": permission denied") {
			t.Skipf("Skipping test (golang.org/issue/19698); unshare failed due to permissions: %s, %v", o, err)
		}
		t.Fatalf("unshare failed: %s, %v", o, err)
	}

	// How do we tell if the namespace was really unshared? It turns out
	// to be simple: just try to remove the executable. If it's still mounted
	// on, the rm will fail. Then we have some cleanup to do:
	// we must force unmount it, then try to remove it again.

	if err := os.Remove(x); err != nil {
		t.Errorf("rm failed on %v: %v", x, err)
		if err := syscall.Unmount(d, syscall.MNT_FORCE); err != nil {
			t.Fatalf("Can't unmount %v: %v", d, err)
		}
		if err := os.Remove(x); err != nil {
			t.Fatalf("rm failed on %v: %v", x, err)
		}
	}

	if err := os.Remove(d); err != nil {
		t.Errorf("rmdir failed on %v: %v", d, err)
	}
}

type capHeader struct {
	version uint32
	pid     int
}

type capData struct {
	effective   uint32
	permitted   uint32
	inheritable uint32
}

const CAP_SYS_TIME = 25

type caps struct {
	hdr  capHeader
	data [2]capData
}

func getCaps() (caps, error) {
	var c caps

	// Get capability version
	if _, _, errno := syscall.Syscall(syscall.SYS_CAPGET, uintptr(unsafe.Pointer(&c.hdr)), uintptr(unsafe.Pointer(nil)), 0); errno != 0 {
		return c, fmt.Errorf("SYS_CAPGET: %v", errno)
	}

	// Get current capabilities
	if _, _, errno := syscall.Syscall(syscall.SYS_CAPGET, uintptr(unsafe.Pointer(&c.hdr)), uintptr(unsafe.Pointer(&c.data[0])), 0); errno != 0 {
		return c, fmt.Errorf("SYS_CAPGET: %v", errno)
	}

	return c, nil
}

func mustSupportAmbientCaps(t *testing.T) {
	var uname syscall.Utsname
	if err := syscall.Uname(&uname); err != nil {
		t.Fatalf("Uname: %v", err)
	}
	var buf [65]byte
	for i, b := range uname.Release {
		buf[i] = byte(b)
	}
	ver := string(buf[:])
	if i := strings.Index(ver, "\x00"); i != -1 {
		ver = ver[:i]
	}
	if strings.HasPrefix(ver, "2.") ||
		strings.HasPrefix(ver, "3.") ||
		strings.HasPrefix(ver, "4.1.") ||
		strings.HasPrefix(ver, "4.2.") {
		t.Skipf("kernel version %q predates required 4.3; skipping test", ver)
	}
}

// TestAmbientCapsHelper isn't a real test. It's used as a helper process for
// TestAmbientCaps.
func TestAmbientCapsHelper(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)

	caps, err := getCaps()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if caps.data[0].effective&(1<<uint(CAP_SYS_TIME)) == 0 {
		fmt.Fprintln(os.Stderr, "CAP_SYS_TIME unexpectedly not in the effective capability mask")
		os.Exit(2)
	}
}

func TestAmbientCaps(t *testing.T) {
	// Make sure we are running as root so we have permissions to use unshare
	// and create a network namespace.
	if os.Getuid() != 0 {
		t.Skip("kernel prohibits unshare in unprivileged process, unless using user namespace")
	}
	mustSupportAmbientCaps(t)

	// When running under the Go continuous build, skip tests for
	// now when under Kubernetes. (where things are root but not quite)
	// Both of these are our own environment variables.
	// See Issue 12815.
	if os.Getenv("GO_BUILDER_NAME") != "" && os.Getenv("IN_KUBERNETES") == "1" {
		t.Skip("skipping test on Kubernetes-based builders; see Issue 12815")
	}

	caps, err := getCaps()
	if err != nil {
		t.Fatal(err)
	}

	// Add CAP_SYS_TIME to the permitted and inheritable capability mask,
	// otherwise we will not be able to add it to the ambient capability mask.
	caps.data[0].permitted |= 1 << uint(CAP_SYS_TIME)
	caps.data[0].inheritable |= 1 << uint(CAP_SYS_TIME)

	if _, _, errno := syscall.Syscall(syscall.SYS_CAPSET, uintptr(unsafe.Pointer(&caps.hdr)), uintptr(unsafe.Pointer(&caps.data[0])), 0); errno != 0 {
		t.Fatalf("SYS_CAPSET: %v", errno)
	}

	u, err := user.Lookup("nobody")
	if err != nil {
		t.Fatal(err)
	}
	uid, err := strconv.ParseInt(u.Uid, 0, 32)
	if err != nil {
		t.Fatal(err)
	}
	gid, err := strconv.ParseInt(u.Gid, 0, 32)
	if err != nil {
		t.Fatal(err)
	}

	// Copy the test binary to a temporary location which is readable by nobody.
	f, err := ioutil.TempFile("", "gotest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()
	e, err := os.Open(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if _, err := io.Copy(f, e); err != nil {
		t.Fatal(err)
	}
	if err := f.Chmod(0755); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(f.Name(), "-test.run=TestAmbientCapsHelper")
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Credential: &syscall.Credential{
			Uid: uint32(uid),
			Gid: uint32(gid),
		},
		AmbientCaps: []uintptr{CAP_SYS_TIME},
	}
	if err := cmd.Run(); err != nil {
		t.Fatal(err.Error())
	}
}
