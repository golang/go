// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux darwin probablyfreebsd probablyopenbsd

package syscall_test

import (
	"flag"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"syscall"
	"testing"
	"time"
)

// TestPassFD tests passing a file descriptor over a Unix socket.
func TestPassFD(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "TestPassFD")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	fds, err := syscall.Socketpair(syscall.AF_LOCAL, syscall.SOCK_STREAM, 0)
	if err != nil {
		t.Fatalf("Socketpair: %v", err)
	}
	defer syscall.Close(fds[0])
	defer syscall.Close(fds[1])
	writeFile := os.NewFile(uintptr(fds[0]), "child-writes")
	readFile := os.NewFile(uintptr(fds[1]), "parent-reads")
	defer writeFile.Close()
	defer readFile.Close()

	cmd := exec.Command(os.Args[0], "-test.run=TestPassFDChild", "--", tempDir)
	cmd.Env = append([]string{"GO_WANT_HELPER_PROCESS=1"}, os.Environ()...)
	cmd.ExtraFiles = []*os.File{writeFile}

	out, err := cmd.CombinedOutput()
	if len(out) > 0 || err != nil {
		t.Errorf("child process: %q, %v", out, err)
		return // not fatalf, so defers above run.
	}

	c, err := net.FileConn(readFile)
	if err != nil {
		t.Errorf("FileConn: %v", err)
		return
	}
	defer c.Close()

	uc, ok := c.(*net.UnixConn)
	if !ok {
		t.Errorf("unexpected FileConn type; expected UnixConn, got %T", c)
		return
	}

	buf := make([]byte, 32) // expect 1 byte
	oob := make([]byte, 32) // expect 24 bytes
	closeUnix := time.AfterFunc(5*time.Second, func() {
		t.Logf("timeout reading from unix socket")
		uc.Close()
	})
	_, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
	closeUnix.Stop()

	scms, err := syscall.ParseSocketControlMessage(oob[:oobn])
	if err != nil {
		t.Errorf("ParseSocketControlMessage: %v", err)
		return
	}
	if len(scms) != 1 {
		t.Errorf("expected 1 SocketControlMessage; got scms = %#v", scms)
		return
	}
	scm := scms[0]
	gotFds, err := syscall.ParseUnixRights(&scm)
	if err != nil {
		t.Errorf("syscall.ParseUnixRights: %v", err)
		return
	}
	if len(gotFds) != 1 {
		t.Errorf("wanted 1 fd; got %#v", gotFds)
		return
	}

	f := os.NewFile(uintptr(gotFds[0]), "fd-from-child")
	defer f.Close()

	got, err := ioutil.ReadAll(f)
	want := "Hello from child process!\n"
	if string(got) != want {
		t.Errorf("child process ReadAll: %q, %v; want %q", got, err, want)
	}
}

// Not a real test. This is the helper child process for TestPassFD.
func TestPassFDChild(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)

	// Look for our fd. I<t should be fd 3, but we work around an fd leak
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
