// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package unix_test

import (
	"flag"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// Tests that below functions, structures and constants are consistent
// on all Unix-like systems.
func _() {
	// program scheduling priority functions and constants
	var (
		_ func(int, int, int) error   = unix.Setpriority
		_ func(int, int) (int, error) = unix.Getpriority
	)
	const (
		_ int = unix.PRIO_USER
		_ int = unix.PRIO_PROCESS
		_ int = unix.PRIO_PGRP
	)

	// termios constants
	const (
		_ int = unix.TCIFLUSH
		_ int = unix.TCIOFLUSH
		_ int = unix.TCOFLUSH
	)

	// fcntl file locking structure and constants
	var (
		_ = unix.Flock_t{
			Type:   int16(0),
			Whence: int16(0),
			Start:  int64(0),
			Len:    int64(0),
			Pid:    int32(0),
		}
	)
	const (
		_ = unix.F_GETLK
		_ = unix.F_SETLK
		_ = unix.F_SETLKW
	)
}

func TestErrnoSignalName(t *testing.T) {
	testErrors := []struct {
		num  syscall.Errno
		name string
	}{
		{syscall.EPERM, "EPERM"},
		{syscall.EINVAL, "EINVAL"},
		{syscall.ENOENT, "ENOENT"},
	}

	for _, te := range testErrors {
		t.Run(fmt.Sprintf("%d/%s", te.num, te.name), func(t *testing.T) {
			e := unix.ErrnoName(te.num)
			if e != te.name {
				t.Errorf("ErrnoName(%d) returned %s, want %s", te.num, e, te.name)
			}
		})
	}

	testSignals := []struct {
		num  syscall.Signal
		name string
	}{
		{syscall.SIGHUP, "SIGHUP"},
		{syscall.SIGPIPE, "SIGPIPE"},
		{syscall.SIGSEGV, "SIGSEGV"},
	}

	for _, ts := range testSignals {
		t.Run(fmt.Sprintf("%d/%s", ts.num, ts.name), func(t *testing.T) {
			s := unix.SignalName(ts.num)
			if s != ts.name {
				t.Errorf("SignalName(%d) returned %s, want %s", ts.num, s, ts.name)
			}
		})
	}
}

func TestFcntlInt(t *testing.T) {
	t.Parallel()
	file, err := ioutil.TempFile("", "TestFnctlInt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(file.Name())
	defer file.Close()
	f := file.Fd()
	flags, err := unix.FcntlInt(f, unix.F_GETFD, 0)
	if err != nil {
		t.Fatal(err)
	}
	if flags&unix.FD_CLOEXEC == 0 {
		t.Errorf("flags %#x do not include FD_CLOEXEC", flags)
	}
}

// TestFcntlFlock tests whether the file locking structure matches
// the calling convention of each kernel.
func TestFcntlFlock(t *testing.T) {
	name := filepath.Join(os.TempDir(), "TestFcntlFlock")
	fd, err := unix.Open(name, unix.O_CREAT|unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer unix.Unlink(name)
	defer unix.Close(fd)
	flock := unix.Flock_t{
		Type:  unix.F_RDLCK,
		Start: 0, Len: 0, Whence: 1,
	}
	if err := unix.FcntlFlock(uintptr(fd), unix.F_GETLK, &flock); err != nil {
		t.Fatalf("FcntlFlock failed: %v", err)
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
	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("cannot exec subprocess on iOS, skipping test")
	}
	if runtime.GOOS == "aix" {
		t.Skip("getsockname issue on AIX 7.2 tl1, skipping test")
	}

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		passFDChild()
		return
	}

	tempDir, err := ioutil.TempDir("", "TestPassFD")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	fds, err := unix.Socketpair(unix.AF_LOCAL, unix.SOCK_STREAM, 0)
	if err != nil {
		t.Fatalf("Socketpair: %v", err)
	}
	defer unix.Close(fds[0])
	defer unix.Close(fds[1])
	writeFile := os.NewFile(uintptr(fds[0]), "child-writes")
	readFile := os.NewFile(uintptr(fds[1]), "parent-reads")
	defer writeFile.Close()
	defer readFile.Close()

	cmd := exec.Command(os.Args[0], "-test.run=^TestPassFD$", "--", tempDir)
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	if lp := os.Getenv("LD_LIBRARY_PATH"); lp != "" {
		cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+lp)
	}
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

	scms, err := unix.ParseSocketControlMessage(oob[:oobn])
	if err != nil {
		t.Fatalf("ParseSocketControlMessage: %v", err)
	}
	if len(scms) != 1 {
		t.Fatalf("expected 1 SocketControlMessage; got scms = %#v", scms)
	}
	scm := scms[0]
	gotFds, err := unix.ParseUnixRights(&scm)
	if err != nil {
		t.Fatalf("unix.ParseUnixRights: %v", err)
	}
	if len(gotFds) != 1 {
		t.Fatalf("wanted 1 fd; got %#v", gotFds)
	}

	f := os.NewFile(uintptr(gotFds[0]), "fd-from-child")
	defer f.Close()

	got, err := ioutil.ReadAll(f)
	want := "Hello from child process!\n"
	if string(got) != want {
		t.Errorf("child process ReadAll: %q, %v; want %q", got, err, want)
	}
}

// passFDChild is the child process used by TestPassFD.
func passFDChild() {
	defer os.Exit(0)

	// Look for our fd. It should be fd 3, but we work around an fd leak
	// bug here (http://golang.org/issue/2603) to let it be elsewhere.
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
	f, err := ioutil.TempFile(tempDir, "")
	if err != nil {
		fmt.Printf("TempFile: %v", err)
		return
	}

	f.Write([]byte("Hello from child process!\n"))
	f.Seek(0, 0)

	rights := unix.UnixRights(int(f.Fd()))
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
			n = len(b) + unix.CmsgLen(4*len(fds))
			b = append(b, unix.UnixRights(fds...)...)
		}
		// Truncate b
		b = b[:n]

		scms, err := unix.ParseSocketControlMessage(b)
		if err != nil {
			t.Fatalf("ParseSocketControlMessage: %v", err)
		}
		if len(scms) != len(testCase) {
			t.Fatalf("expected %v SocketControlMessage; got scms = %#v", len(testCase), scms)
		}
		for i, scm := range scms {
			gotFds, err := unix.ParseUnixRights(&scm)
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

func TestRlimit(t *testing.T) {
	var rlimit, zero unix.Rlimit
	err := unix.Getrlimit(unix.RLIMIT_NOFILE, &rlimit)
	if err != nil {
		t.Fatalf("Getrlimit: save failed: %v", err)
	}
	if zero == rlimit {
		t.Fatalf("Getrlimit: save failed: got zero value %#v", rlimit)
	}
	set := rlimit
	set.Cur = set.Max - 1
	if runtime.GOOS == "darwin" && set.Cur > 10240 {
		// The max file limit is 10240, even though
		// the max returned by Getrlimit is 1<<63-1.
		// This is OPEN_MAX in sys/syslimits.h.
		set.Cur = 10240
	}
	err = unix.Setrlimit(unix.RLIMIT_NOFILE, &set)
	if err != nil {
		t.Fatalf("Setrlimit: set failed: %#v %v", set, err)
	}
	var get unix.Rlimit
	err = unix.Getrlimit(unix.RLIMIT_NOFILE, &get)
	if err != nil {
		t.Fatalf("Getrlimit: get failed: %v", err)
	}
	set = rlimit
	set.Cur = set.Max - 1
	if set != get {
		// Seems like Darwin requires some privilege to
		// increase the soft limit of rlimit sandbox, though
		// Setrlimit never reports an error.
		switch runtime.GOOS {
		case "darwin":
		default:
			t.Fatalf("Rlimit: change failed: wanted %#v got %#v", set, get)
		}
	}
	err = unix.Setrlimit(unix.RLIMIT_NOFILE, &rlimit)
	if err != nil {
		t.Fatalf("Setrlimit: restore failed: %#v %v", rlimit, err)
	}
}

func TestSeekFailure(t *testing.T) {
	_, err := unix.Seek(-1, 0, 0)
	if err == nil {
		t.Fatalf("Seek(-1, 0, 0) did not fail")
	}
	str := err.Error() // used to crash on Linux
	t.Logf("Seek: %v", str)
	if str == "" {
		t.Fatalf("Seek(-1, 0, 0) return error with empty message")
	}
}

func TestDup(t *testing.T) {
	file, err := ioutil.TempFile("", "TestDup")
	if err != nil {
		t.Fatalf("Tempfile failed: %v", err)
	}
	defer os.Remove(file.Name())
	defer file.Close()
	f := int(file.Fd())

	newFd, err := unix.Dup(f)
	if err != nil {
		t.Fatalf("Dup: %v", err)
	}

	err = unix.Dup2(newFd, newFd+1)
	if err != nil {
		t.Fatalf("Dup2: %v", err)
	}

	b1 := []byte("Test123")
	b2 := make([]byte, 7)
	_, err = unix.Write(newFd+1, b1)
	if err != nil {
		t.Fatalf("Write to dup2 fd failed: %v", err)
	}
	_, err = unix.Seek(f, 0, 0)
	if err != nil {
		t.Fatalf("Seek failed: %v", err)
	}
	_, err = unix.Read(f, b2)
	if err != nil {
		t.Fatalf("Read back failed: %v", err)
	}
	if string(b1) != string(b2) {
		t.Errorf("Dup: stdout write not in file, expected %v, got %v", string(b1), string(b2))
	}
}

func TestPoll(t *testing.T) {
	if runtime.GOOS == "android" ||
		(runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64")) {
		t.Skip("mkfifo syscall is not available on android and iOS, skipping test")
	}

	f, cleanup := mktmpfifo(t)
	defer cleanup()

	const timeout = 100

	ok := make(chan bool, 1)
	go func() {
		select {
		case <-time.After(10 * timeout * time.Millisecond):
			t.Errorf("Poll: failed to timeout after %d milliseconds", 10*timeout)
		case <-ok:
		}
	}()

	fds := []unix.PollFd{{Fd: int32(f.Fd()), Events: unix.POLLIN}}
	n, err := unix.Poll(fds, timeout)
	ok <- true
	if err != nil {
		t.Errorf("Poll: unexpected error: %v", err)
		return
	}
	if n != 0 {
		t.Errorf("Poll: wrong number of events: got %v, expected %v", n, 0)
		return
	}
}

func TestGetwd(t *testing.T) {
	fd, err := os.Open(".")
	if err != nil {
		t.Fatalf("Open .: %s", err)
	}
	defer fd.Close()
	// Directory list for test. Do not worry if any are symlinks or do not
	// exist on some common unix desktop environments. That will be checked.
	dirs := []string{"/", "/usr/bin", "/etc", "/var", "/opt"}
	switch runtime.GOOS {
	case "android":
		dirs = []string{"/", "/system/bin"}
	case "darwin":
		switch runtime.GOARCH {
		case "arm", "arm64":
			d1, err := ioutil.TempDir("", "d1")
			if err != nil {
				t.Fatalf("TempDir: %v", err)
			}
			d2, err := ioutil.TempDir("", "d2")
			if err != nil {
				t.Fatalf("TempDir: %v", err)
			}
			dirs = []string{d1, d2}
		}
	}
	oldwd := os.Getenv("PWD")
	for _, d := range dirs {
		// Check whether d exists, is a dir and that d's path does not contain a symlink
		fi, err := os.Stat(d)
		if err != nil || !fi.IsDir() {
			t.Logf("Test dir %s stat error (%v) or not a directory, skipping", d, err)
			continue
		}
		check, err := filepath.EvalSymlinks(d)
		if err != nil || check != d {
			t.Logf("Test dir %s (%s) is symlink or other error (%v), skipping", d, check, err)
			continue
		}
		err = os.Chdir(d)
		if err != nil {
			t.Fatalf("Chdir: %v", err)
		}
		pwd, err := unix.Getwd()
		if err != nil {
			t.Fatalf("Getwd in %s: %s", d, err)
		}
		os.Setenv("PWD", oldwd)
		err = fd.Chdir()
		if err != nil {
			// We changed the current directory and cannot go back.
			// Don't let the tests continue; they'll scribble
			// all over some other directory.
			fmt.Fprintf(os.Stderr, "fchdir back to dot failed: %s\n", err)
			os.Exit(1)
		}
		if pwd != d {
			t.Fatalf("Getwd returned %q want %q", pwd, d)
		}
	}
}

func TestFstatat(t *testing.T) {
	defer chtmpdir(t)()

	touch(t, "file1")

	var st1 unix.Stat_t
	err := unix.Stat("file1", &st1)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}

	var st2 unix.Stat_t
	err = unix.Fstatat(unix.AT_FDCWD, "file1", &st2, 0)
	if err != nil {
		t.Fatalf("Fstatat: %v", err)
	}

	if st1 != st2 {
		t.Errorf("Fstatat: returned stat does not match Stat")
	}

	err = os.Symlink("file1", "symlink1")
	if err != nil {
		t.Fatal(err)
	}

	err = unix.Lstat("symlink1", &st1)
	if err != nil {
		t.Fatalf("Lstat: %v", err)
	}

	err = unix.Fstatat(unix.AT_FDCWD, "symlink1", &st2, unix.AT_SYMLINK_NOFOLLOW)
	if err != nil {
		t.Fatalf("Fstatat: %v", err)
	}

	if st1 != st2 {
		t.Errorf("Fstatat: returned stat does not match Lstat")
	}
}

func TestFchmodat(t *testing.T) {
	defer chtmpdir(t)()

	touch(t, "file1")
	err := os.Symlink("file1", "symlink1")
	if err != nil {
		t.Fatal(err)
	}

	mode := os.FileMode(0444)
	err = unix.Fchmodat(unix.AT_FDCWD, "symlink1", uint32(mode), 0)
	if err != nil {
		t.Fatalf("Fchmodat: unexpected error: %v", err)
	}

	fi, err := os.Stat("file1")
	if err != nil {
		t.Fatal(err)
	}

	if fi.Mode() != mode {
		t.Errorf("Fchmodat: failed to change file mode: expected %v, got %v", mode, fi.Mode())
	}

	mode = os.FileMode(0644)
	didChmodSymlink := true
	err = unix.Fchmodat(unix.AT_FDCWD, "symlink1", uint32(mode), unix.AT_SYMLINK_NOFOLLOW)
	if err != nil {
		if (runtime.GOOS == "android" || runtime.GOOS == "linux" || runtime.GOOS == "solaris") && err == unix.EOPNOTSUPP {
			// Linux and Illumos don't support flags != 0
			didChmodSymlink = false
		} else {
			t.Fatalf("Fchmodat: unexpected error: %v", err)
		}
	}

	if !didChmodSymlink {
		// Didn't change mode of the symlink. On Linux, the permissions
		// of a symbolic link are always 0777 according to symlink(7)
		mode = os.FileMode(0777)
	}

	var st unix.Stat_t
	err = unix.Lstat("symlink1", &st)
	if err != nil {
		t.Fatal(err)
	}

	got := os.FileMode(st.Mode & 0777)
	if got != mode {
		t.Errorf("Fchmodat: failed to change symlink mode: expected %v, got %v", mode, got)
	}
}

func TestMkdev(t *testing.T) {
	major := uint32(42)
	minor := uint32(7)
	dev := unix.Mkdev(major, minor)

	if unix.Major(dev) != major {
		t.Errorf("Major(%#x) == %d, want %d", dev, unix.Major(dev), major)
	}
	if unix.Minor(dev) != minor {
		t.Errorf("Minor(%#x) == %d, want %d", dev, unix.Minor(dev), minor)
	}
}

// mktmpfifo creates a temporary FIFO and provides a cleanup function.
func mktmpfifo(t *testing.T) (*os.File, func()) {
	err := unix.Mkfifo("fifo", 0666)
	if err != nil {
		t.Fatalf("mktmpfifo: failed to create FIFO: %v", err)
	}

	f, err := os.OpenFile("fifo", os.O_RDWR, 0666)
	if err != nil {
		os.Remove("fifo")
		t.Fatalf("mktmpfifo: failed to open FIFO: %v", err)
	}

	return f, func() {
		f.Close()
		os.Remove("fifo")
	}
}

// utilities taken from os/os_test.go

func touch(t *testing.T, name string) {
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

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
