// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

// chtmpdir changes the working directory to a new temporary directory and
// provides a cleanup function. Used when PWD is read-only.
func chtmpdir(t *testing.T) func() {
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	d, err := ioutil.TempDir("", "test")
	if err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	if err := os.Chdir(d); err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	return func() {
		if err := os.Chdir(oldwd); err != nil {
			t.Fatalf("chtmpdir: %v", err)
		}
		os.RemoveAll(d)
	}
}

func touch(t *testing.T, name string) {
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

const (
	_AT_SYMLINK_NOFOLLOW = 0x100
	_AT_FDCWD            = -0x64
	_AT_EACCESS          = 0x200
	_F_OK                = 0
	_R_OK                = 4
)

func TestFaccessat(t *testing.T) {
	defer chtmpdir(t)()
	touch(t, "file1")

	err := syscall.Faccessat(_AT_FDCWD, "file1", _R_OK, 0)
	if err != nil {
		t.Errorf("Faccessat: unexpected error: %v", err)
	}

	err = syscall.Faccessat(_AT_FDCWD, "file1", _R_OK, 2)
	if err != syscall.EINVAL {
		t.Errorf("Faccessat: unexpected error: %v, want EINVAL", err)
	}

	err = syscall.Faccessat(_AT_FDCWD, "file1", _R_OK, _AT_EACCESS)
	if err != nil {
		t.Errorf("Faccessat: unexpected error: %v", err)
	}

	err = os.Symlink("file1", "symlink1")
	if err != nil {
		t.Fatal(err)
	}

	err = syscall.Faccessat(_AT_FDCWD, "symlink1", _R_OK, _AT_SYMLINK_NOFOLLOW)
	if err != nil {
		t.Errorf("Faccessat SYMLINK_NOFOLLOW: unexpected error %v", err)
	}

	// We can't really test _AT_SYMLINK_NOFOLLOW, because there
	// doesn't seem to be any way to change the mode of a symlink.
	// We don't test _AT_EACCESS because such tests are only
	// meaningful if run as root.

	err = syscall.Fchmodat(_AT_FDCWD, "file1", 0, 0)
	if err != nil {
		t.Errorf("Fchmodat: unexpected error %v", err)
	}

	err = syscall.Faccessat(_AT_FDCWD, "file1", _F_OK, _AT_SYMLINK_NOFOLLOW)
	if err != nil {
		t.Errorf("Faccessat: unexpected error: %v", err)
	}

	err = syscall.Faccessat(_AT_FDCWD, "file1", _R_OK, _AT_SYMLINK_NOFOLLOW)
	if err != syscall.EACCES {
		if syscall.Getuid() != 0 {
			t.Errorf("Faccessat: unexpected error: %v, want EACCES", err)
		}
	}
}

func TestFchmodat(t *testing.T) {
	defer chtmpdir(t)()

	touch(t, "file1")
	os.Symlink("file1", "symlink1")

	err := syscall.Fchmodat(_AT_FDCWD, "symlink1", 0444, 0)
	if err != nil {
		t.Fatalf("Fchmodat: unexpected error: %v", err)
	}

	fi, err := os.Stat("file1")
	if err != nil {
		t.Fatal(err)
	}

	if fi.Mode() != 0444 {
		t.Errorf("Fchmodat: failed to change mode: expected %v, got %v", 0444, fi.Mode())
	}

	err = syscall.Fchmodat(_AT_FDCWD, "symlink1", 0444, _AT_SYMLINK_NOFOLLOW)
	if err != syscall.EOPNOTSUPP {
		t.Fatalf("Fchmodat: unexpected error: %v, expected EOPNOTSUPP", err)
	}
}

func TestMain(m *testing.M) {
	if os.Getenv("GO_DEATHSIG_PARENT") == "1" {
		deathSignalParent()
	} else if os.Getenv("GO_DEATHSIG_CHILD") == "1" {
		deathSignalChild()
	} else if os.Getenv("GO_SYSCALL_NOERROR") == "1" {
		syscallNoError()
	}

	os.Exit(m.Run())
}

func TestLinuxDeathSignal(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("skipping root only test")
	}

	// Copy the test binary to a location that a non-root user can read/execute
	// after we drop privileges
	tempDir, err := ioutil.TempDir("", "TestDeathSignal")
	if err != nil {
		t.Fatalf("cannot create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	os.Chmod(tempDir, 0755)

	tmpBinary := filepath.Join(tempDir, filepath.Base(os.Args[0]))

	src, err := os.Open(os.Args[0])
	if err != nil {
		t.Fatalf("cannot open binary %q, %v", os.Args[0], err)
	}
	defer src.Close()

	dst, err := os.OpenFile(tmpBinary, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		t.Fatalf("cannot create temporary binary %q, %v", tmpBinary, err)
	}
	if _, err := io.Copy(dst, src); err != nil {
		t.Fatalf("failed to copy test binary to %q, %v", tmpBinary, err)
	}
	err = dst.Close()
	if err != nil {
		t.Fatalf("failed to close test binary %q, %v", tmpBinary, err)
	}

	cmd := exec.Command(tmpBinary)
	cmd.Env = []string{"GO_DEATHSIG_PARENT=1"}
	chldStdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("failed to create new stdin pipe: %v", err)
	}
	chldStdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("failed to create new stdout pipe: %v", err)
	}
	cmd.Stderr = os.Stderr

	err = cmd.Start()
	defer cmd.Wait()
	if err != nil {
		t.Fatalf("failed to start first child process: %v", err)
	}

	chldPipe := bufio.NewReader(chldStdout)

	if got, err := chldPipe.ReadString('\n'); got == "start\n" {
		syscall.Kill(cmd.Process.Pid, syscall.SIGTERM)

		go func() {
			time.Sleep(5 * time.Second)
			chldStdin.Close()
		}()

		want := "ok\n"
		if got, err = chldPipe.ReadString('\n'); got != want {
			t.Fatalf("expected %q, received %q, %v", want, got, err)
		}
	} else {
		t.Fatalf("did not receive start from child, received %q, %v", got, err)
	}
}

func deathSignalParent() {
	cmd := exec.Command(os.Args[0])
	cmd.Env = []string{"GO_DEATHSIG_CHILD=1"}
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	attrs := syscall.SysProcAttr{
		Pdeathsig: syscall.SIGUSR1,
		// UID/GID 99 is the user/group "nobody" on RHEL/Fedora and is
		// unused on Ubuntu
		Credential: &syscall.Credential{Uid: 99, Gid: 99},
	}
	cmd.SysProcAttr = &attrs

	err := cmd.Start()
	if err != nil {
		fmt.Fprintf(os.Stderr, "death signal parent error: %v\n", err)
		os.Exit(1)
	}
	cmd.Wait()
	os.Exit(0)
}

func deathSignalChild() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGUSR1)
	go func() {
		<-c
		fmt.Println("ok")
		os.Exit(0)
	}()
	fmt.Println("start")

	buf := make([]byte, 32)
	os.Stdin.Read(buf)

	// We expected to be signaled before stdin closed
	fmt.Println("not ok")
	os.Exit(1)
}

func TestParseNetlinkMessage(t *testing.T) {
	for i, b := range [][]byte{
		{103, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 11, 0, 1, 0, 0, 0, 0, 5, 8, 0, 3,
			0, 8, 0, 6, 0, 0, 0, 0, 1, 63, 0, 10, 0, 69, 16, 0, 59, 39, 82, 64, 0, 64, 6, 21, 89, 127, 0, 0,
			1, 127, 0, 0, 1, 230, 228, 31, 144, 32, 186, 155, 211, 185, 151, 209, 179, 128, 24, 1, 86,
			53, 119, 0, 0, 1, 1, 8, 10, 0, 17, 234, 12, 0, 17, 189, 126, 107, 106, 108, 107, 106, 13, 10,
		},
		{106, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 11, 0, 1, 0, 0, 0, 0, 3, 8, 0, 3,
			0, 8, 0, 6, 0, 0, 0, 0, 1, 66, 0, 10, 0, 69, 0, 0, 62, 230, 255, 64, 0, 64, 6, 85, 184, 127, 0, 0,
			1, 127, 0, 0, 1, 237, 206, 31, 144, 73, 197, 128, 65, 250, 60, 192, 97, 128, 24, 1, 86, 253, 21, 0,
			0, 1, 1, 8, 10, 0, 51, 106, 89, 0, 51, 102, 198, 108, 104, 106, 108, 107, 104, 108, 107, 104, 10,
		},
		{102, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 11, 0, 1, 0, 0, 0, 0, 1, 8, 0, 3, 0,
			8, 0, 6, 0, 0, 0, 0, 1, 62, 0, 10, 0, 69, 0, 0, 58, 231, 2, 64, 0, 64, 6, 85, 185, 127, 0, 0, 1, 127,
			0, 0, 1, 237, 206, 31, 144, 73, 197, 128, 86, 250, 60, 192, 97, 128, 24, 1, 86, 104, 64, 0, 0, 1, 1, 8,
			10, 0, 52, 198, 200, 0, 51, 135, 232, 101, 115, 97, 103, 103, 10,
		},
	} {
		m, err := syscall.ParseNetlinkMessage(b)
		if err != syscall.EINVAL {
			t.Errorf("#%d: got %v; want EINVAL", i, err)
		}
		if m != nil {
			t.Errorf("#%d: got %v; want nil", i, m)
		}
	}
}

func TestSyscallNoError(t *testing.T) {
	// On Linux there are currently no syscalls which don't fail and return
	// a value larger than 0xfffffffffffff001 so we could test RawSyscall
	// vs. RawSyscallNoError on 64bit architectures.
	if unsafe.Sizeof(uintptr(0)) != 4 {
		t.Skip("skipping on non-32bit architecture")
	}

	// See https://golang.org/issue/35422
	// On MIPS, Linux returns whether the syscall had an error in a separate
	// register (R7), not using a negative return value as on other
	// architectures.
	if runtime.GOARCH == "mips" || runtime.GOARCH == "mipsle" {
		t.Skipf("skipping on %s", runtime.GOARCH)
	}

	if os.Getuid() != 0 {
		t.Skip("skipping root only test")
	}

	if runtime.GOOS == "android" {
		t.Skip("skipping on rooted android, see issue 27364")
	}

	// Copy the test binary to a location that a non-root user can read/execute
	// after we drop privileges
	tempDir, err := ioutil.TempDir("", "TestSyscallNoError")
	if err != nil {
		t.Fatalf("cannot create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)
	os.Chmod(tempDir, 0755)

	tmpBinary := filepath.Join(tempDir, filepath.Base(os.Args[0]))

	src, err := os.Open(os.Args[0])
	if err != nil {
		t.Fatalf("cannot open binary %q, %v", os.Args[0], err)
	}
	defer src.Close()

	dst, err := os.OpenFile(tmpBinary, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		t.Fatalf("cannot create temporary binary %q, %v", tmpBinary, err)
	}
	if _, err := io.Copy(dst, src); err != nil {
		t.Fatalf("failed to copy test binary to %q, %v", tmpBinary, err)
	}
	err = dst.Close()
	if err != nil {
		t.Fatalf("failed to close test binary %q, %v", tmpBinary, err)
	}

	uid := uint32(0xfffffffe)
	err = os.Chown(tmpBinary, int(uid), -1)
	if err != nil {
		t.Fatalf("failed to chown test binary %q, %v", tmpBinary, err)
	}

	err = os.Chmod(tmpBinary, 0755|os.ModeSetuid)
	if err != nil {
		t.Fatalf("failed to set setuid bit on test binary %q, %v", tmpBinary, err)
	}

	cmd := exec.Command(tmpBinary)
	cmd.Env = []string{"GO_SYSCALL_NOERROR=1"}

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to start first child process: %v", err)
	}

	got := strings.TrimSpace(string(out))
	want := strconv.FormatUint(uint64(uid)+1, 10) + " / " +
		strconv.FormatUint(uint64(-uid), 10) + " / " +
		strconv.FormatUint(uint64(uid), 10)
	if got != want {
		if filesystemIsNoSUID(tmpBinary) {
			t.Skip("skipping test when temp dir is mounted nosuid")
		}
		// formatted so the values are aligned for easier comparison
		t.Errorf("expected %s,\ngot      %s", want, got)
	}
}

// filesystemIsNoSUID reports whether the filesystem for the given
// path is mounted nosuid.
func filesystemIsNoSUID(path string) bool {
	var st syscall.Statfs_t
	if syscall.Statfs(path, &st) != nil {
		return false
	}
	return st.Flags&syscall.MS_NOSUID != 0
}

func syscallNoError() {
	// Test that the return value from SYS_GETEUID32 (which cannot fail)
	// doesn't get treated as an error (see https://golang.org/issue/22924)
	euid1, _, e := syscall.RawSyscall(syscall.Sys_GETEUID, 0, 0, 0)
	euid2, _ := syscall.RawSyscallNoError(syscall.Sys_GETEUID, 0, 0, 0)

	fmt.Println(uintptr(euid1), "/", int(e), "/", uintptr(euid2))
	os.Exit(0)
}
