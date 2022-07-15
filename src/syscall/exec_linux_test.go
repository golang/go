// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package syscall_test

import (
	"bytes"
	"flag"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"os/user"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

func isDocker() bool {
	_, err := os.Stat("/.dockerenv")
	return err == nil
}

func isLXC() bool {
	return os.Getenv("container") == "lxc"
}

func skipInContainer(t *testing.T) {
	// TODO: the callers of this func are using this func to skip
	// tests when running as some sort of "fake root" that's uid 0
	// but lacks certain Linux capabilities. Most of the Go builds
	// run in privileged containers, though, where root is much
	// closer (if not identical) to the real root. We should test
	// for what we need exactly (which capabilities are active?),
	// instead of just assuming "docker == bad". Then we'd get more test
	// coverage on a bunch of builders too.
	if isDocker() {
		t.Skip("skip this test in Docker container")
	}
	if isLXC() {
		t.Skip("skip this test in LXC container")
	}
}

func skipNoUserNamespaces(t *testing.T) {
	if _, err := os.Stat("/proc/self/ns/user"); err != nil {
		if os.IsNotExist(err) {
			t.Skip("kernel doesn't support user namespaces")
		}
		if os.IsPermission(err) {
			t.Skip("unable to test user namespaces due to permissions")
		}
		t.Fatalf("Failed to stat /proc/self/ns/user: %v", err)
	}
}

func skipUnprivilegedUserClone(t *testing.T) {
	// Skip the test if the sysctl that prevents unprivileged user
	// from creating user namespaces is enabled.
	data, errRead := os.ReadFile("/proc/sys/kernel/unprivileged_userns_clone")
	if os.IsNotExist(errRead) {
		// This file is only available in some Debian/Ubuntu kernels.
		return
	}
	if errRead != nil || len(data) < 1 || data[0] == '0' {
		t.Skip("kernel prohibits user namespace in unprivileged process")
	}
}

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
	skipInContainer(t)
	skipNoUserNamespaces(t)
	if isChrooted(t) {
		// create_user_ns in the kernel (see
		// https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/kernel/user_namespace.c)
		// forbids the creation of user namespaces when chrooted.
		t.Skip("cannot create user namespaces when chrooted")
	}
	// On some systems, there is a sysctl setting.
	if os.Getuid() != 0 {
		skipUnprivilegedUserClone(t)
	}
	// On Centos 7 make sure they set the kernel parameter user_namespace=1
	// See issue 16283 and 20796.
	if _, err := os.Stat("/sys/module/user_namespace/parameters/enable"); err == nil {
		buf, _ := os.ReadFile("/sys/module/user_namespace/parameters/enabled")
		if !strings.HasPrefix(string(buf), "Y") {
			t.Skip("kernel doesn't support user namespaces")
		}
	}

	// On Centos 7.5+, user namespaces are disabled if user.max_user_namespaces = 0
	if _, err := os.Stat("/proc/sys/user/max_user_namespaces"); err == nil {
		buf, errRead := os.ReadFile("/proc/sys/user/max_user_namespaces")
		if errRead == nil && buf[0] == '0' {
			t.Skip("kernel doesn't support user namespaces")
		}
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
	testNEWUSERRemap(t, 0, 0, true)
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
	skipInContainer(t)
	// Make sure we are running as root so we have permissions to use unshare
	// and create a network namespace.
	if os.Getuid() != 0 {
		t.Skip("kernel prohibits unshare in unprivileged process, unless using user namespace")
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

	orig, err := os.ReadFile(path)
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
	t.Logf("id: %s", strOut)

	expected := "uid=0(root) gid=0(root)"
	// Just check prefix because some distros reportedly output a
	// context parameter; see https://golang.org/issue/16224.
	// Alpine does not output groups; see https://golang.org/issue/19938.
	if !strings.HasPrefix(strOut, expected) {
		t.Errorf("expected prefix: %q", expected)
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
	t.Logf("id: %s", strOut)

	// As in TestGroupCleanup, just check prefix.
	// The actual groups and contexts seem to vary from one distro to the next.
	expected := "uid=0(root) gid=0(root) groups=0(root)"
	if !strings.HasPrefix(strOut, expected) {
		t.Errorf("expected prefix: %q", expected)
	}
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
	skipInContainer(t)
	// Make sure we are running as root so we have permissions to use unshare
	// and create a network namespace.
	if os.Getuid() != 0 {
		t.Skip("kernel prohibits unshare in unprivileged process, unless using user namespace")
	}

	d, err := os.MkdirTemp("", "unshare")
	if err != nil {
		t.Fatalf("tempdir: %v", err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestUnshareMountNameSpaceHelper", d)
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
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
	skipInContainer(t)
	// Make sure we are running as root so we have permissions to use unshare
	// and create a network namespace.
	if os.Getuid() != 0 {
		t.Skip("kernel prohibits unshare in unprivileged process, unless using user namespace")
	}

	d, err := os.MkdirTemp("", "unshare")
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
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
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

func TestUnshareUidGidMappingHelper(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)
	if err := syscall.Chroot(os.TempDir()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
}

// Test for Issue 29789: unshare fails when uid/gid mapping is specified
func TestUnshareUidGidMapping(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("test exercises unprivileged user namespace, fails with privileges")
	}
	checkUserNS(t)
	cmd := exec.Command(os.Args[0], "-test.run=TestUnshareUidGidMappingHelper")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Unshareflags:               syscall.CLONE_NEWNS | syscall.CLONE_NEWUSER,
		GidMappingsEnableSetgroups: false,
		UidMappings: []syscall.SysProcIDMap{
			{
				ContainerID: 0,
				HostID:      syscall.Getuid(),
				Size:        1,
			},
		},
		GidMappings: []syscall.SysProcIDMap{
			{
				ContainerID: 0,
				HostID:      syscall.Getgid(),
				Size:        1,
			},
		},
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Cmd failed with err %v, output: %s", err, out)
	}
}

func prepareCgroupFD(t *testing.T) (int, string) {
	t.Helper()

	const O_PATH = 0x200000 // Same for all architectures, but for some reason not defined in syscall for 386||amd64.

	// Requires cgroup v2.
	const prefix = "/sys/fs/cgroup"
	selfCg, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		if os.IsNotExist(err) || os.IsPermission(err) {
			t.Skip(err)
		}
		t.Fatal(err)
	}

	// Expect a single line like this:
	// 0::/user.slice/user-1000.slice/user@1000.service/app.slice/vte-spawn-891992a2-efbb-4f28-aedb-b24f9e706770.scope
	// Otherwise it's either cgroup v1 or a hybrid hierarchy.
	if bytes.Count(selfCg, []byte("\n")) > 1 {
		t.Skip("cgroup v2 not available")
	}
	cg := bytes.TrimPrefix(selfCg, []byte("0::"))
	if len(cg) == len(selfCg) { // No prefix found.
		t.Skipf("cgroup v2 not available (/proc/self/cgroup contents: %q)", selfCg)
	}

	// Need clone3 with CLONE_INTO_CGROUP support.
	_, err = syscall.ForkExec("non-existent binary", nil, &syscall.ProcAttr{
		Sys: &syscall.SysProcAttr{
			UseCgroupFD: true,
			CgroupFD:    -1,
		},
	})
	// // EPERM can be returned if clone3 is not enabled by seccomp.
	if err == syscall.ENOSYS || err == syscall.EPERM {
		t.Skipf("clone3 with CLONE_INTO_CGROUP not available: %v", err)
	}

	// Need an ability to create a sub-cgroup.
	subCgroup, err := os.MkdirTemp(prefix+string(bytes.TrimSpace(cg)), "subcg-")
	if err != nil {
		if os.IsPermission(err) {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	t.Cleanup(func() { syscall.Rmdir(subCgroup) })

	cgroupFD, err := syscall.Open(subCgroup, O_PATH, 0)
	if err != nil {
		t.Fatal(&os.PathError{Op: "open", Path: subCgroup, Err: err})
	}
	t.Cleanup(func() { syscall.Close(cgroupFD) })

	return cgroupFD, "/" + path.Base(subCgroup)
}

func TestUseCgroupFD(t *testing.T) {
	fd, suffix := prepareCgroupFD(t)

	cmd := exec.Command(os.Args[0], "-test.run=TestUseCgroupFDHelper")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		UseCgroupFD: true,
		CgroupFD:    fd,
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Cmd failed with err %v, output: %s", err, out)
	}
	// NB: this wouldn't work with cgroupns.
	if !bytes.HasSuffix(bytes.TrimSpace(out), []byte(suffix)) {
		t.Fatalf("got: %q, want: a line that ends with %q", out, suffix)
	}
}

func TestUseCgroupFDHelper(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)
	// Read and print own cgroup path.
	selfCg, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Print(string(selfCg))
}

type capHeader struct {
	version uint32
	pid     int32
}

type capData struct {
	effective   uint32
	permitted   uint32
	inheritable uint32
}

const CAP_SYS_TIME = 25
const CAP_SYSLOG = 34

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
	ver, _, _ = strings.Cut(ver, "\x00")
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
	if caps.data[1].effective&(1<<uint(CAP_SYSLOG&31)) == 0 {
		fmt.Fprintln(os.Stderr, "CAP_SYSLOG unexpectedly not in the effective capability mask")
		os.Exit(2)
	}
}

func TestAmbientCaps(t *testing.T) {
	// Make sure we are running as root so we have permissions to use unshare
	// and create a network namespace.
	if os.Getuid() != 0 {
		t.Skip("kernel prohibits unshare in unprivileged process, unless using user namespace")
	}

	testAmbientCaps(t, false)
}

func TestAmbientCapsUserns(t *testing.T) {
	checkUserNS(t)
	testAmbientCaps(t, true)
}

func testAmbientCaps(t *testing.T, userns bool) {
	skipInContainer(t)
	mustSupportAmbientCaps(t)

	skipUnprivilegedUserClone(t)

	// skip on android, due to lack of lookup support
	if runtime.GOOS == "android" {
		t.Skip("skipping test on android; see Issue 27327")
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
	f, err := os.CreateTemp("", "gotest")
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
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Credential: &syscall.Credential{
			Uid: uint32(uid),
			Gid: uint32(gid),
		},
		AmbientCaps: []uintptr{CAP_SYS_TIME, CAP_SYSLOG},
	}
	if userns {
		cmd.SysProcAttr.Cloneflags = syscall.CLONE_NEWUSER
		const nobody = 65534
		uid := os.Getuid()
		gid := os.Getgid()
		cmd.SysProcAttr.UidMappings = []syscall.SysProcIDMap{{
			ContainerID: int(nobody),
			HostID:      int(uid),
			Size:        int(1),
		}}
		cmd.SysProcAttr.GidMappings = []syscall.SysProcIDMap{{
			ContainerID: int(nobody),
			HostID:      int(gid),
			Size:        int(1),
		}}

		// Set credentials to run as user and group nobody.
		cmd.SysProcAttr.Credential = &syscall.Credential{
			Uid: nobody,
			Gid: nobody,
		}
	}
	if err := cmd.Run(); err != nil {
		t.Fatal(err.Error())
	}
}
