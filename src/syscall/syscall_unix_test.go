// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package syscall_test

import (
	"flag"
	"fmt"
	"internal/testenv"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"syscall"
	"testing"
	"time"
)

// Tests that below functions, structures and constants are consistent
// on all Unix-like systems.
func _() {
	// program scheduling priority functions and constants
	var (
		_ func(int, int, int) error   = syscall.Setpriority
		_ func(int, int) (int, error) = syscall.Getpriority
	)
	const (
		_ int = syscall.PRIO_USER
		_ int = syscall.PRIO_PROCESS
		_ int = syscall.PRIO_PGRP
	)

	// termios constants
	const (
		_ int = syscall.TCIFLUSH
		_ int = syscall.TCIOFLUSH
		_ int = syscall.TCOFLUSH
	)

	// fcntl file locking structure and constants
	var (
		_ = syscall.Flock_t{
			Type:   int16(0),
			Whence: int16(0),
			Start:  int64(0),
			Len:    int64(0),
			Pid:    int32(0),
		}
	)
	const (
		_ = syscall.F_GETLK
		_ = syscall.F_SETLK
		_ = syscall.F_SETLKW
	)
}

// TestFcntlFlock tests whether the file locking structure matches
// the calling convention of each kernel.
// On some Linux systems, glibc uses another set of values for the
// commands and translates them to the correct value that the kernel
// expects just before the actual fcntl syscall. As Go uses raw
// syscalls directly, it must use the real value, not the glibc value.
// Thus this test also verifies that the Flock_t structure can be
// roundtripped with F_SETLK and F_GETLK.
func TestFcntlFlock(t *testing.T) {
	if runtime.GOOS == "ios" {
		t.Skip("skipping; no child processes allowed on iOS")
	}
	flock := syscall.Flock_t{
		Type:  syscall.F_WRLCK,
		Start: 31415, Len: 271828, Whence: 1,
	}
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "" {
		// parent
		tempDir := t.TempDir()
		name := filepath.Join(tempDir, "TestFcntlFlock")
		fd, err := syscall.Open(name, syscall.O_CREAT|syscall.O_RDWR|syscall.O_CLOEXEC, 0)
		if err != nil {
			t.Fatalf("Open failed: %v", err)
		}
		// f takes ownership of fd, and will close it.
		//
		// N.B. This defer is also necessary to keep f alive
		// while we use its fd, preventing its finalizer from
		// executing.
		f := os.NewFile(uintptr(fd), name)
		defer f.Close()

		if err := syscall.Ftruncate(int(f.Fd()), 1<<20); err != nil {
			t.Fatalf("Ftruncate(1<<20) failed: %v", err)
		}
		if err := syscall.FcntlFlock(f.Fd(), syscall.F_SETLK, &flock); err != nil {
			t.Fatalf("FcntlFlock(F_SETLK) failed: %v", err)
		}

		cmd := exec.Command(os.Args[0], "-test.run=^TestFcntlFlock$")
		cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
		cmd.ExtraFiles = []*os.File{f}
		out, err := cmd.CombinedOutput()
		if len(out) > 0 || err != nil {
			t.Fatalf("child process: %q, %v", out, err)
		}
	} else {
		// child
		got := flock
		// make sure the child lock is conflicting with the parent lock
		got.Start--
		got.Len++
		if err := syscall.FcntlFlock(3, syscall.F_GETLK, &got); err != nil {
			t.Fatalf("FcntlFlock(F_GETLK) failed: %v", err)
		}
		flock.Pid = int32(syscall.Getppid())
		// Linux kernel always set Whence to 0
		flock.Whence = 0
		if got.Type == flock.Type && got.Start == flock.Start && got.Len == flock.Len && got.Pid == flock.Pid && got.Whence == flock.Whence {
			os.Exit(0)
		}
		t.Fatalf("FcntlFlock got %v, want %v", got, flock)
	}
}

// TestPassFD tests passing a file descriptor over a Unix socket.
//
// This test involved both a parent and child process. The parent
// process is invoked as a normal test, with "go test", which then
// runs the child process by running the current test binary with args
// "-test.run=^TestPassFD$" and an environment variable used to signal
// that the test should become the child process instead.
func TestPassFD(t *testing.T) {
	testenv.MustHaveExec(t)

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		passFDChild()
		return
	}

	if runtime.GOOS == "aix" {
		// Unix network isn't properly working on AIX 7.2 with Technical Level < 2
		out, err := exec.Command("oslevel", "-s").Output()
		if err != nil {
			t.Skipf("skipping on AIX because oslevel -s failed: %v", err)
		}
		if len(out) < len("7200-XX-ZZ-YYMM") { // AIX 7.2, Tech Level XX, Service Pack ZZ, date YYMM
			t.Skip("skipping on AIX because oslevel -s hasn't the right length")
		}
		aixVer := string(out[:4])
		tl, err := strconv.Atoi(string(out[5:7]))
		if err != nil {
			t.Skipf("skipping on AIX because oslevel -s output cannot be parsed: %v", err)
		}
		if aixVer < "7200" || (aixVer == "7200" && tl < 2) {
			t.Skip("skipped on AIX versions previous to 7.2 TL 2")
		}

	}

	tempDir := t.TempDir()

	fds, err := syscall.Socketpair(syscall.AF_LOCAL, syscall.SOCK_STREAM, 0)
	if err != nil {
		t.Fatalf("Socketpair: %v", err)
	}
	writeFile := os.NewFile(uintptr(fds[0]), "child-writes")
	readFile := os.NewFile(uintptr(fds[1]), "parent-reads")
	defer writeFile.Close()
	defer readFile.Close()

	cmd := exec.Command(os.Args[0], "-test.run=^TestPassFD$", "--", tempDir)
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	cmd.ExtraFiles = []*os.File{writeFile}

	out, err := cmd.CombinedOutput()
	if len(out) > 0 || err != nil {
		t.Fatalf("child process: %q, %v", out, err)
	}

	c, err := net.FileConn(readFile)
	if err != nil {
		t.Fatalf("FileConn: %v", err)
	}
	defer c.Close()

	uc, ok := c.(*net.UnixConn)
	if !ok {
		t.Fatalf("unexpected FileConn type; expected UnixConn, got %T", c)
	}

	buf := make([]byte, 32) // expect 1 byte
	oob := make([]byte, 32) // expect 24 bytes
	closeUnix := time.AfterFunc(5*time.Second, func() {
		t.Logf("timeout reading from unix socket")
		uc.Close()
	})
	_, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
	if err != nil {
		t.Fatalf("ReadMsgUnix: %v", err)
	}
	closeUnix.Stop()

	scms, err := syscall.ParseSocketControlMessage(oob[:oobn])
	if err != nil {
		t.Fatalf("ParseSocketControlMessage: %v", err)
	}
	if len(scms) != 1 {
		t.Fatalf("expected 1 SocketControlMessage; got scms = %#v", scms)
	}
	scm := scms[0]
	gotFds, err := syscall.ParseUnixRights(&scm)
	if err != nil {
		t.Fatalf("syscall.ParseUnixRights: %v", err)
	}
	if len(gotFds) != 1 {
		t.Fatalf("wanted 1 fd; got %#v", gotFds)
	}

	f := os.NewFile(uintptr(gotFds[0]), "fd-from-child")
	defer f.Close()

	got, err := io.ReadAll(f)
	want := "Hello from child process!\n"
	if string(got) != want {
		t.Errorf("child process ReadAll: %q, %v; want %q", got, err, want)
	}
}

// passFDChild is the child process used by TestPassFD.
func passFDChild() {
	defer os.Exit(0)

	// Look for our fd. It should be fd 3, but we work around an fd leak
	// bug here (https://golang.org/issue/2603) to let it be elsewhere.
	var uc *net.UnixConn
	for fd := uintptr(3); fd <= 10; fd++ {
		f := os.NewFile(fd, "unix-conn")
		var ok bool
		netc, _ := net.FileConn(f)
		uc, ok = netc.(*net.UnixConn)
		if ok {
			break
		}
	}
	if uc == nil {
		fmt.Println("failed to find unix fd")
		return
	}

	// Make a file f to send to our parent process on uc.
	// We make it in tempDir, which our parent will clean up.
	flag.Parse()
	tempDir := flag.Arg(0)
	f, err := os.CreateTemp(tempDir, "")
	if err != nil {
		fmt.Printf("TempFile: %v", err)
		return
	}
	// N.B. This defer is also necessary to keep f alive
	// while we use its fd, preventing its finalizer from
	// executing.
	defer f.Close()

	f.Write([]byte("Hello from child process!\n"))
	f.Seek(0, io.SeekStart)

	rights := syscall.UnixRights(int(f.Fd()))
	dummyByte := []byte("x")
	n, oobn, err := uc.WriteMsgUnix(dummyByte, rights, nil)
	if err != nil {
		fmt.Printf("WriteMsgUnix: %v", err)
		return
	}
	if n != 1 || oobn != len(rights) {
		fmt.Printf("WriteMsgUnix = %d, %d; want 1, %d", n, oobn, len(rights))
		return
	}
}

// TestUnixRightsRoundtrip tests that UnixRights, ParseSocketControlMessage,
// and ParseUnixRights are able to successfully round-trip lists of file descriptors.
func TestUnixRightsRoundtrip(t *testing.T) {
	testCases := [...][][]int{
		{{42}},
		{{1, 2}},
		{{3, 4, 5}},
		{{}},
		{{1, 2}, {3, 4, 5}, {}, {7}},
	}
	for _, testCase := range testCases {
		b := []byte{}
		var n int
		for _, fds := range testCase {
			// Last assignment to n wins
			n = len(b) + syscall.CmsgLen(4*len(fds))
			b = append(b, syscall.UnixRights(fds...)...)
		}
		// Truncate b
		b = b[:n]

		scms, err := syscall.ParseSocketControlMessage(b)
		if err != nil {
			t.Fatalf("ParseSocketControlMessage: %v", err)
		}
		if len(scms) != len(testCase) {
			t.Fatalf("expected %v SocketControlMessage; got scms = %#v", len(testCase), scms)
		}
		for i, scm := range scms {
			gotFds, err := syscall.ParseUnixRights(&scm)
			if err != nil {
				t.Fatalf("ParseUnixRights: %v", err)
			}
			wantFds := testCase[i]
			if len(gotFds) != len(wantFds) {
				t.Fatalf("expected %v fds, got %#v", len(wantFds), gotFds)
			}
			for j, fd := range gotFds {
				if fd != wantFds[j] {
					t.Fatalf("expected fd %v, got %v", wantFds[j], fd)
				}
			}
		}
	}
}

func TestSeekFailure(t *testing.T) {
	_, err := syscall.Seek(-1, 0, io.SeekStart)
	if err == nil {
		t.Fatalf("Seek(-1, 0, 0) did not fail")
	}
	str := err.Error() // used to crash on Linux
	t.Logf("Seek: %v", str)
	if str == "" {
		t.Fatalf("Seek(-1, 0, 0) return error with empty message")
	}
}

func TestSetsockoptString(t *testing.T) {
	// should not panic on empty string, see issue #31277
	err := syscall.SetsockoptString(-1, 0, 0, "")
	if err == nil {
		t.Fatalf("SetsockoptString: did not fail")
	}
}

func TestENFILETemporary(t *testing.T) {
	if !syscall.ENFILE.Temporary() {
		t.Error("ENFILE is not treated as a temporary error")
	}
}
