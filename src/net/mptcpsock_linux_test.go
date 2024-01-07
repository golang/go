// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"context"
	"errors"
	"syscall"
	"testing"
)

func newLocalListenerMPTCP(t *testing.T, envVar bool) Listener {
	lc := &ListenConfig{}

	if envVar {
		if !lc.MultipathTCP() {
			t.Fatal("MultipathTCP Listen is not on despite GODEBUG=multipathtcp=1")
		}
	} else {
		if lc.MultipathTCP() {
			t.Error("MultipathTCP should be off by default")
		}

		lc.SetMultipathTCP(true)
		if !lc.MultipathTCP() {
			t.Fatal("MultipathTCP is not on after having been forced to on")
		}
	}

	ln, err := lc.Listen(context.Background(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	return ln
}

func postAcceptMPTCP(ls *localServer, ch chan<- error) {
	defer close(ch)

	if len(ls.cl) == 0 {
		ch <- errors.New("no accepted stream")
		return
	}

	c := ls.cl[0]

	tcp, ok := c.(*TCPConn)
	if !ok {
		ch <- errors.New("struct is not a TCPConn")
		return
	}

	mptcp, err := tcp.MultipathTCP()
	if err != nil {
		ch <- err
		return
	}

	if !mptcp {
		ch <- errors.New("incoming connection is not with MPTCP")
		return
	}

	// Also check the method for the older kernels if not tested before
	if hasSOLMPTCP && !isUsingMPTCPProto(tcp.fd) {
		ch <- errors.New("incoming connection is not an MPTCP proto")
		return
	}
}

func dialerMPTCP(t *testing.T, addr string, envVar bool) {
	d := &Dialer{}

	if envVar {
		if !d.MultipathTCP() {
			t.Fatal("MultipathTCP Dialer is not on despite GODEBUG=multipathtcp=1")
		}
	} else {
		if d.MultipathTCP() {
			t.Error("MultipathTCP should be off by default")
		}

		d.SetMultipathTCP(true)
		if !d.MultipathTCP() {
			t.Fatal("MultipathTCP is not on after having been forced to on")
		}
	}

	c, err := d.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	tcp, ok := c.(*TCPConn)
	if !ok {
		t.Fatal("struct is not a TCPConn")
	}

	// Transfer a bit of data to make sure everything is still OK
	snt := []byte("MPTCP TEST")
	if _, err := c.Write(snt); err != nil {
		t.Fatal(err)
	}
	b := make([]byte, len(snt))
	if _, err := c.Read(b); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(snt, b) {
		t.Errorf("sent bytes (%s) are different from received ones (%s)", snt, b)
	}

	mptcp, err := tcp.MultipathTCP()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("outgoing connection from %s with mptcp: %t", addr, mptcp)

	if !mptcp {
		t.Error("outgoing connection is not with MPTCP")
	}

	// Also check the method for the older kernels if not tested before
	if hasSOLMPTCP && !isUsingMPTCPProto(tcp.fd) {
		t.Error("outgoing connection is not an MPTCP proto")
	}
}

func canCreateMPTCPSocket() bool {
	// We want to know if we can create an MPTCP socket, not just if it is
	// available (mptcpAvailable()): it could be blocked by the admin
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM, _IPPROTO_MPTCP)
	if err != nil {
		return false
	}

	syscall.Close(fd)
	return true
}

func testMultiPathTCP(t *testing.T, envVar bool) {
	if envVar {
		t.Log("Test with GODEBUG=multipathtcp=1")
		t.Setenv("GODEBUG", "multipathtcp=1")
	} else {
		t.Log("Test with GODEBUG=multipathtcp=0")
		t.Setenv("GODEBUG", "multipathtcp=0")
	}

	ln := newLocalListenerMPTCP(t, envVar)

	// similar to tcpsock_test:TestIPv6LinkLocalUnicastTCP
	ls := (&streamListener{Listener: ln}).newLocalServer()
	defer ls.teardown()

	if g, w := ls.Listener.Addr().Network(), "tcp"; g != w {
		t.Fatalf("Network type mismatch: got %q, want %q", g, w)
	}

	genericCh := make(chan error)
	mptcpCh := make(chan error)
	handler := func(ls *localServer, ln Listener) {
		ls.transponder(ln, genericCh)
		postAcceptMPTCP(ls, mptcpCh)
	}
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	dialerMPTCP(t, ln.Addr().String(), envVar)

	if err := <-genericCh; err != nil {
		t.Error(err)
	}
	if err := <-mptcpCh; err != nil {
		t.Error(err)
	}
}

func TestMultiPathTCP(t *testing.T) {
	if !canCreateMPTCPSocket() {
		t.Skip("Cannot create MPTCP sockets")
	}

	for _, envVar := range []bool{false, true} {
		testMultiPathTCP(t, envVar)
	}
}
