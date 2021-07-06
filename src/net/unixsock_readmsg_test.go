// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"os"
	"syscall"
	"testing"
	"time"
)

func TestUnixConnReadMsgUnixSCMRightsCloseOnExec(t *testing.T) {
	if !testableNetwork("unix") {
		t.Skip("not unix system")
	}

	scmFile, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatalf("file open: %v", err)
	}
	defer scmFile.Close()

	rights := syscall.UnixRights(int(scmFile.Fd()))
	fds, err := syscall.Socketpair(syscall.AF_LOCAL, syscall.SOCK_STREAM, 0)
	if err != nil {
		t.Fatalf("Socketpair: %v", err)
	}

	writeFile := os.NewFile(uintptr(fds[0]), "write-socket")
	defer writeFile.Close()
	readFile := os.NewFile(uintptr(fds[1]), "read-socket")
	defer readFile.Close()

	cw, err := FileConn(writeFile)
	if err != nil {
		t.Fatalf("FileConn: %v", err)
	}
	defer cw.Close()
	cr, err := FileConn(readFile)
	if err != nil {
		t.Fatalf("FileConn: %v", err)
	}
	defer cr.Close()

	ucw, ok := cw.(*UnixConn)
	if !ok {
		t.Fatalf("got %T; want UnixConn", cw)
	}
	ucr, ok := cr.(*UnixConn)
	if !ok {
		t.Fatalf("got %T; want UnixConn", cr)
	}

	oob := make([]byte, syscall.CmsgSpace(4))
	err = ucw.SetWriteDeadline(time.Now().Add(5 * time.Second))
	if err != nil {
		t.Fatalf("Can't set unix connection timeout: %v", err)
	}
	_, _, err = ucw.WriteMsgUnix(nil, rights, nil)
	if err != nil {
		t.Fatalf("UnixConn readMsg: %v", err)
	}
	err = ucr.SetReadDeadline(time.Now().Add(5 * time.Second))
	if err != nil {
		t.Fatalf("Can't set unix connection timeout: %v", err)
	}
	_, oobn, _, _, err := ucr.ReadMsgUnix(nil, oob)
	if err != nil {
		t.Fatalf("UnixConn readMsg: %v", err)
	}

	scms, err := syscall.ParseSocketControlMessage(oob[:oobn])
	if err != nil {
		t.Fatalf("ParseSocketControlMessage: %v", err)
	}
	if len(scms) != 1 {
		t.Fatalf("got scms = %#v; expected 1 SocketControlMessage", scms)
	}
	scm := scms[0]
	gotFDs, err := syscall.ParseUnixRights(&scm)
	if err != nil {
		t.Fatalf("syscall.ParseUnixRights: %v", err)
	}
	if len(gotFDs) != 1 {
		t.Fatalf("got FDs %#v: wanted only 1 fd", gotFDs)
	}
	defer func() {
		if err := syscall.Close(gotFDs[0]); err != nil {
			t.Fatalf("fail to close gotFDs: %v", err)
		}
	}()

	flags, err := fcntl(gotFDs[0], syscall.F_GETFD, 0)
	if err != nil {
		t.Fatalf("Can't get flags of fd:%#v, with err:%v", gotFDs[0], err)
	}
	if flags&syscall.FD_CLOEXEC == 0 {
		t.Fatalf("got flags %#x, want %#x (FD_CLOEXEC) set", flags, syscall.FD_CLOEXEC)
	}
}
