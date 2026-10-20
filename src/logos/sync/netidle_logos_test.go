// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package sync_test

import (
	"net"
	"runtime"
	"syscall"
	"testing"
	"time"
)

const (
	idleConnCount      = 10000
	idleConnFdSlack    = 1024
	idleConnHeapPerMax = 1024
	idleConnDialTO     = 10 * time.Second
	idleConnAcceptTO   = 60 * time.Second
)

// TestLogosIdleConnectionsHeapGrowth pins SPEC Sec 17 #21: the build uses
// standard netpoll with no per-connection pinned buffer, so ten thousand
// idle loopback TCP connections grow the heap by less than 1 KB each.
// It skips under -short and when the file limit cannot cover both ends.
func TestLogosIdleConnectionsHeapGrowth(t *testing.T) {
	if testing.Short() {
		t.Skip("needs 10000 idle loopback TCP connections; run without -short")
	}
	ensureIdleConnFileLimit(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen loopback: %v", err)
	}
	clients := make([]net.Conn, 0, idleConnCount)
	servers := make([]net.Conn, 0, idleConnCount)
	defer func() {
		_ = ln.Close()
		closeIdleConns(clients)
		closeIdleConns(servers)
	}()
	accepted := make(chan net.Conn, idleConnCount)
	acceptErr := make(chan error, 1)
	go acceptIdleConns(ln, idleConnCount, accepted, acceptErr)
	before := heapAllocAfterGC()
	clients = dialIdleConns(t, ln.Addr().String(), idleConnCount)
	servers = collectIdleConns(t, accepted, acceptErr, idleConnCount)
	after := heapAllocAfterGC()
	growth := int64(after) - int64(before)
	per := float64(growth) / float64(idleConnCount)
	t.Logf("idle conns: heap %d -> %d (growth %d, %.1f per conn)", before, after, growth, per)
	if growth >= int64(idleConnCount*idleConnHeapPerMax) {
		t.Fatalf("heap grew %d bytes for %d idle conns (%.1f each), want < %d each", growth, idleConnCount, per, idleConnHeapPerMax)
	}
}

// ensureIdleConnFileLimit raises the soft file limit to cover both ends
// of every idle connection, skipping when the hard limit forbids it.
func ensureIdleConnFileLimit(t *testing.T) {
	t.Helper()
	need := uint64(2*idleConnCount + idleConnFdSlack)
	var lim syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &lim); err != nil {
		t.Skipf("getrlimit: %v", err)
	}
	if lim.Cur >= need {
		return
	}
	if lim.Max < need {
		t.Skipf("RLIMIT_NOFILE hard %d covers fewer than %d fds", lim.Max, need)
	}
	lim.Cur = need
	if err := syscall.Setrlimit(syscall.RLIMIT_NOFILE, &lim); err != nil {
		t.Skipf("setrlimit: %v", err)
	}
}

// acceptIdleConns accepts n connections, forwarding them to accepted.
func acceptIdleConns(ln net.Listener, n int, accepted chan<- net.Conn, acceptErr chan<- error) {
	for range n {
		c, err := ln.Accept()
		if err != nil {
			acceptErr <- err
			return
		}
		accepted <- c
	}
}

// dialIdleConns opens n loopback TCP clients to addr.
func dialIdleConns(t *testing.T, addr string, n int) []net.Conn {
	t.Helper()
	conns := make([]net.Conn, 0, n)
	for range n {
		c, err := net.DialTimeout("tcp", addr, idleConnDialTO)
		if err != nil {
			closeIdleConns(conns)
			t.Fatalf("dial %d: %v", len(conns), err)
		}
		conns = append(conns, c)
	}
	return conns
}

// collectIdleConns gathers the n server ends accepted by acceptIdleConns.
func collectIdleConns(t *testing.T, accepted <-chan net.Conn, acceptErr <-chan error, n int) []net.Conn {
	t.Helper()
	conns := make([]net.Conn, 0, n)
	timeout := time.After(idleConnAcceptTO)
	for range n {
		select {
		case c := <-accepted:
			conns = append(conns, c)
		case err := <-acceptErr:
			closeIdleConns(conns)
			t.Fatalf("accept: %v", err)
		case <-timeout:
			closeIdleConns(conns)
			t.Fatalf("accepted %d of %d conns before timeout", len(conns), n)
		}
	}
	return conns
}

// heapAllocAfterGC collects twice and reports live heap bytes.
func heapAllocAfterGC() uint64 {
	runtime.GC()
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

// closeIdleConns closes every connection, ignoring close errors.
func closeIdleConns(conns []net.Conn) {
	for _, c := range conns {
		_ = c.Close()
	}
}
