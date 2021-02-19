// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"bufio"
	"fmt"
	"io"
	"io/fs"
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
	d, err := os.MkdirTemp("", "test")
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
	tempDir, err := os.MkdirTemp("", "TestDeathSignal")
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
	cmd.Env = append(os.Environ(), "GO_DEATHSIG_PARENT=1")
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
	cmd.Env = append(os.Environ(),
		"GO_DEATHSIG_PARENT=",
		"GO_DEATHSIG_CHILD=1",
	)
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
	tempDir, err := os.MkdirTemp("", "TestSyscallNoError")
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

	err = os.Chmod(tmpBinary, 0755|fs.ModeSetuid)
	if err != nil {
		t.Fatalf("failed to set setuid bit on test binary %q, %v", tmpBinary, err)
	}

	cmd := exec.Command(tmpBinary)
	cmd.Env = append(os.Environ(), "GO_SYSCALL_NOERROR=1")

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

// reference uapi/linux/prctl.h
const (
	PR_GET_KEEPCAPS uintptr = 7
	PR_SET_KEEPCAPS         = 8
)

// TestAllThreadsSyscall tests that the go runtime can perform
// syscalls that execute on all OSThreads - with which to support
// POSIX semantics for security state changes.
func TestAllThreadsSyscall(t *testing.T) {
	if _, _, err := syscall.AllThreadsSyscall(syscall.SYS_PRCTL, PR_SET_KEEPCAPS, 0, 0); err == syscall.ENOTSUP {
		t.Skip("AllThreadsSyscall disabled with cgo")
	}

	fns := []struct {
		label string
		fn    func(uintptr) error
	}{
		{
			label: "prctl<3-args>",
			fn: func(v uintptr) error {
				_, _, e := syscall.AllThreadsSyscall(syscall.SYS_PRCTL, PR_SET_KEEPCAPS, v, 0)
				if e != 0 {
					return e
				}
				return nil
			},
		},
		{
			label: "prctl<6-args>",
			fn: func(v uintptr) error {
				_, _, e := syscall.AllThreadsSyscall6(syscall.SYS_PRCTL, PR_SET_KEEPCAPS, v, 0, 0, 0, 0)
				if e != 0 {
					return e
				}
				return nil
			},
		},
	}

	waiter := func(q <-chan uintptr, r chan<- uintptr, once bool) {
		for x := range q {
			runtime.LockOSThread()
			v, _, e := syscall.Syscall(syscall.SYS_PRCTL, PR_GET_KEEPCAPS, 0, 0)
			if e != 0 {
				t.Errorf("tid=%d prctl(PR_GET_KEEPCAPS) failed: %v", syscall.Gettid(), e)
			} else if x != v {
				t.Errorf("tid=%d prctl(PR_GET_KEEPCAPS) mismatch: got=%d want=%d", syscall.Gettid(), v, x)
			}
			r <- v
			if once {
				break
			}
			runtime.UnlockOSThread()
		}
	}

	// launches per fns member.
	const launches = 11
	question := make(chan uintptr)
	response := make(chan uintptr)
	defer close(question)

	routines := 0
	for i, v := range fns {
		for j := 0; j < launches; j++ {
			// Add another goroutine - the closest thing
			// we can do to encourage more OS thread
			// creation - while the test is running.  The
			// actual thread creation may or may not be
			// needed, based on the number of available
			// unlocked OS threads at the time waiter
			// calls runtime.LockOSThread(), but the goal
			// of doing this every time through the loop
			// is to race thread creation with v.fn(want)
			// being executed. Via the once boolean we
			// also encourage one in 5 waiters to return
			// locked after participating in only one
			// question response sequence. This allows the
			// test to race thread destruction too.
			once := routines%5 == 4
			go waiter(question, response, once)

			// Keep a count of how many goroutines are
			// going to participate in the
			// question/response test. This will count up
			// towards 2*launches minus the count of
			// routines that have been invoked with
			// once=true.
			routines++

			// Decide what value we want to set the
			// process-shared KEEPCAPS. Note, there is
			// an explicit repeat of 0 when we change the
			// variant of the syscall being used.
			want := uintptr(j & 1)

			// Invoke the AllThreadsSyscall* variant.
			if err := v.fn(want); err != nil {
				t.Errorf("[%d,%d] %s(PR_SET_KEEPCAPS, %d, ...): %v", i, j, v.label, j&1, err)
			}

			// At this point, we want all launched Go
			// routines to confirm that they see the
			// wanted value for KEEPCAPS.
			for k := 0; k < routines; k++ {
				question <- want
			}

			// At this point, we should have a large
			// number of locked OS threads all wanting to
			// reply.
			for k := 0; k < routines; k++ {
				if got := <-response; got != want {
					t.Errorf("[%d,%d,%d] waiter result got=%d, want=%d", i, j, k, got, want)
				}
			}

			// Provide an explicit opportunity for this Go
			// routine to change Ms.
			runtime.Gosched()

			if once {
				// One waiter routine will have exited.
				routines--
			}

			// Whatever M we are now running on, confirm
			// we see the wanted value too.
			if v, _, e := syscall.Syscall(syscall.SYS_PRCTL, PR_GET_KEEPCAPS, 0, 0); e != 0 {
				t.Errorf("[%d,%d] prctl(PR_GET_KEEPCAPS) failed: %v", i, j, e)
			} else if v != want {
				t.Errorf("[%d,%d] prctl(PR_GET_KEEPCAPS) gave wrong value: got=%v, want=1", i, j, v)
			}
		}
	}
}

// compareStatus is used to confirm the contents of the thread
// specific status files match expectations.
func compareStatus(filter, expect string) error {
	expected := filter + expect
	pid := syscall.Getpid()
	fs, err := os.ReadDir(fmt.Sprintf("/proc/%d/task", pid))
	if err != nil {
		return fmt.Errorf("unable to find %d tasks: %v", pid, err)
	}
	expectedProc := fmt.Sprintf("Pid:\t%d", pid)
	foundAThread := false
	for _, f := range fs {
		tf := fmt.Sprintf("/proc/%s/status", f.Name())
		d, err := os.ReadFile(tf)
		if err != nil {
			// There are a surprising number of ways this
			// can error out on linux.  We've seen all of
			// the following, so treat any error here as
			// equivalent to the "process is gone":
			//    os.IsNotExist(err),
			//    "... : no such process",
			//    "... : bad file descriptor.
			continue
		}
		lines := strings.Split(string(d), "\n")
		for _, line := range lines {
			// Different kernel vintages pad differently.
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "Pid:\t") {
				// On loaded systems, it is possible
				// for a TID to be reused really
				// quickly. As such, we need to
				// validate that the thread status
				// info we just read is a task of the
				// same process PID as we are
				// currently running, and not a
				// recently terminated thread
				// resurfaced in a different process.
				if line != expectedProc {
					break
				}
				// Fall through in the unlikely case
				// that filter at some point is
				// "Pid:\t".
			}
			if strings.HasPrefix(line, filter) {
				if line != expected {
					return fmt.Errorf("%q got:%q want:%q (bad) [pid=%d file:'%s' %v]\n", tf, line, expected, pid, string(d), expectedProc)
				}
				foundAThread = true
				break
			}
		}
	}
	if !foundAThread {
		return fmt.Errorf("found no thread /proc/<TID>/status files for process %q", expectedProc)
	}
	return nil
}

// killAThread locks the goroutine to an OS thread and exits; this
// causes an OS thread to terminate.
func killAThread(c <-chan struct{}) {
	runtime.LockOSThread()
	<-c
	return
}

// TestSetuidEtc performs tests on all of the wrapped system calls
// that mirror to the 9 glibc syscalls with POSIX semantics. The test
// here is considered authoritative and should compile and run
// CGO_ENABLED=0 or 1. Note, there is an extended copy of this same
// test in ../../misc/cgo/test/issue1435.go which requires
// CGO_ENABLED=1 and launches pthreads from C that run concurrently
// with the Go code of the test - and the test validates that these
// pthreads are also kept in sync with the security state changed with
// the syscalls. Care should be taken to mirror any enhancements to
// this test here in that file too.
func TestSetuidEtc(t *testing.T) {
	if syscall.Getuid() != 0 {
		t.Skip("skipping root only test")
	}
	vs := []struct {
		call           string
		fn             func() error
		filter, expect string
	}{
		{call: "Setegid(1)", fn: func() error { return syscall.Setegid(1) }, filter: "Gid:", expect: "\t0\t1\t0\t1"},
		{call: "Setegid(0)", fn: func() error { return syscall.Setegid(0) }, filter: "Gid:", expect: "\t0\t0\t0\t0"},

		{call: "Seteuid(1)", fn: func() error { return syscall.Seteuid(1) }, filter: "Uid:", expect: "\t0\t1\t0\t1"},
		{call: "Setuid(0)", fn: func() error { return syscall.Setuid(0) }, filter: "Uid:", expect: "\t0\t0\t0\t0"},

		{call: "Setgid(1)", fn: func() error { return syscall.Setgid(1) }, filter: "Gid:", expect: "\t1\t1\t1\t1"},
		{call: "Setgid(0)", fn: func() error { return syscall.Setgid(0) }, filter: "Gid:", expect: "\t0\t0\t0\t0"},

		{call: "Setgroups([]int{0,1,2,3})", fn: func() error { return syscall.Setgroups([]int{0, 1, 2, 3}) }, filter: "Groups:", expect: "\t0 1 2 3"},
		{call: "Setgroups(nil)", fn: func() error { return syscall.Setgroups(nil) }, filter: "Groups:", expect: ""},
		{call: "Setgroups([]int{0})", fn: func() error { return syscall.Setgroups([]int{0}) }, filter: "Groups:", expect: "\t0"},

		{call: "Setregid(101,0)", fn: func() error { return syscall.Setregid(101, 0) }, filter: "Gid:", expect: "\t101\t0\t0\t0"},
		{call: "Setregid(0,102)", fn: func() error { return syscall.Setregid(0, 102) }, filter: "Gid:", expect: "\t0\t102\t102\t102"},
		{call: "Setregid(0,0)", fn: func() error { return syscall.Setregid(0, 0) }, filter: "Gid:", expect: "\t0\t0\t0\t0"},

		{call: "Setreuid(1,0)", fn: func() error { return syscall.Setreuid(1, 0) }, filter: "Uid:", expect: "\t1\t0\t0\t0"},
		{call: "Setreuid(0,2)", fn: func() error { return syscall.Setreuid(0, 2) }, filter: "Uid:", expect: "\t0\t2\t2\t2"},
		{call: "Setreuid(0,0)", fn: func() error { return syscall.Setreuid(0, 0) }, filter: "Uid:", expect: "\t0\t0\t0\t0"},

		{call: "Setresgid(101,0,102)", fn: func() error { return syscall.Setresgid(101, 0, 102) }, filter: "Gid:", expect: "\t101\t0\t102\t0"},
		{call: "Setresgid(0,102,101)", fn: func() error { return syscall.Setresgid(0, 102, 101) }, filter: "Gid:", expect: "\t0\t102\t101\t102"},
		{call: "Setresgid(0,0,0)", fn: func() error { return syscall.Setresgid(0, 0, 0) }, filter: "Gid:", expect: "\t0\t0\t0\t0"},

		{call: "Setresuid(1,0,2)", fn: func() error { return syscall.Setresuid(1, 0, 2) }, filter: "Uid:", expect: "\t1\t0\t2\t0"},
		{call: "Setresuid(0,2,1)", fn: func() error { return syscall.Setresuid(0, 2, 1) }, filter: "Uid:", expect: "\t0\t2\t1\t2"},
		{call: "Setresuid(0,0,0)", fn: func() error { return syscall.Setresuid(0, 0, 0) }, filter: "Uid:", expect: "\t0\t0\t0\t0"},
	}

	for i, v := range vs {
		// Generate some thread churn as we execute the tests.
		c := make(chan struct{})
		go killAThread(c)
		close(c)

		if err := v.fn(); err != nil {
			t.Errorf("[%d] %q failed: %v", i, v.call, err)
			continue
		}
		if err := compareStatus(v.filter, v.expect); err != nil {
			t.Errorf("[%d] %q comparison: %v", i, v.call, err)
		}
	}
}
