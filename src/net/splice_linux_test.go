// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package net

import (
	"internal/poll"
	"io"
	"os"
	"strconv"
	"sync"
	"syscall"
	"testing"
)

func TestSplice(t *testing.T) {
	t.Run("tcp-to-tcp", func(t *testing.T) { testSplice(t, "tcp", "tcp") })
	if !testableNetwork("unixgram") {
		t.Skip("skipping unix-to-tcp tests")
	}
	t.Run("unix-to-tcp", func(t *testing.T) { testSplice(t, "unix", "tcp") })
	t.Run("tcp-to-unix", func(t *testing.T) { testSplice(t, "tcp", "unix") })
	t.Run("tcp-to-file", func(t *testing.T) { testSpliceToFile(t, "tcp", "file") })
	t.Run("unix-to-file", func(t *testing.T) { testSpliceToFile(t, "unix", "file") })
	t.Run("no-unixpacket", testSpliceNoUnixpacket)
	t.Run("no-unixgram", testSpliceNoUnixgram)
}

func testSpliceToFile(t *testing.T, upNet, downNet string) {
	t.Run("simple", spliceTestCase{upNet, downNet, 128, 128, 0}.testFile)
	t.Run("multipleWrite", spliceTestCase{upNet, downNet, 4096, 1 << 20, 0}.testFile)
	t.Run("big", spliceTestCase{upNet, downNet, 5 << 20, 1 << 30, 0}.testFile)
	t.Run("honorsLimitedReader", spliceTestCase{upNet, downNet, 4096, 1 << 20, 1 << 10}.testFile)
	t.Run("updatesLimitedReaderN", spliceTestCase{upNet, downNet, 1024, 4096, 4096 + 100}.testFile)
	t.Run("limitedReaderAtLimit", spliceTestCase{upNet, downNet, 32, 128, 128}.testFile)
}

func testSplice(t *testing.T, upNet, downNet string) {
	t.Run("simple", spliceTestCase{upNet, downNet, 128, 128, 0}.test)
	t.Run("multipleWrite", spliceTestCase{upNet, downNet, 4096, 1 << 20, 0}.test)
	t.Run("big", spliceTestCase{upNet, downNet, 5 << 20, 1 << 30, 0}.test)
	t.Run("honorsLimitedReader", spliceTestCase{upNet, downNet, 4096, 1 << 20, 1 << 10}.test)
	t.Run("updatesLimitedReaderN", spliceTestCase{upNet, downNet, 1024, 4096, 4096 + 100}.test)
	t.Run("limitedReaderAtLimit", spliceTestCase{upNet, downNet, 32, 128, 128}.test)
	t.Run("readerAtEOF", func(t *testing.T) { testSpliceReaderAtEOF(t, upNet, downNet) })
	t.Run("issue25985", func(t *testing.T) { testSpliceIssue25985(t, upNet, downNet) })
}

type spliceTestCase struct {
	upNet, downNet string

	chunkSize, totalSize int
	limitReadSize        int
}

func (tc spliceTestCase) test(t *testing.T) {
	hook := hookSplice(t)

	// We need to use the actual size for startTestSocketPeer when testing with LimitedReader,
	// otherwise the child process created in startTestSocketPeer will hang infinitely because of
	// the mismatch of data size to transfer.
	size := tc.totalSize
	if tc.limitReadSize > 0 {
		if tc.limitReadSize < size {
			size = tc.limitReadSize
		}
	}

	clientUp, serverUp := spawnTestSocketPair(t, tc.upNet)
	defer serverUp.Close()
	cleanup, err := startTestSocketPeer(t, clientUp, "w", tc.chunkSize, size)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup(t)
	clientDown, serverDown := spawnTestSocketPair(t, tc.downNet)
	defer serverDown.Close()
	cleanup, err = startTestSocketPeer(t, clientDown, "r", tc.chunkSize, size)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup(t)

	var r io.Reader = serverUp
	if tc.limitReadSize > 0 {
		r = &io.LimitedReader{
			N: int64(tc.limitReadSize),
			R: serverUp,
		}
		defer serverUp.Close()
	}
	n, err := io.Copy(serverDown, r)
	if err != nil {
		t.Fatal(err)
	}

	if want := int64(size); want != n {
		t.Errorf("want %d bytes spliced, got %d", want, n)
	}

	if tc.limitReadSize > 0 {
		wantN := 0
		if tc.limitReadSize > size {
			wantN = tc.limitReadSize - size
		}

		if n := r.(*io.LimitedReader).N; n != int64(wantN) {
			t.Errorf("r.N = %d, want %d", n, wantN)
		}
	}

	// poll.Splice is expected to be called when the source is not
	// a wrapper or the destination is TCPConn.
	if tc.limitReadSize == 0 || tc.downNet == "tcp" {
		// We should have called poll.Splice with the right file descriptor arguments.
		if n > 0 && !hook.called {
			t.Fatal("expected poll.Splice to be called")
		}

		verifySpliceFds(t, serverDown, hook, "dst")
		verifySpliceFds(t, serverUp, hook, "src")

		// poll.Splice is expected to handle the data transmission successfully.
		if !hook.handled || hook.written != int64(size) || hook.err != nil {
			t.Errorf("expected handled = true, written = %d, err = nil, but got handled = %t, written = %d, err = %v",
				size, hook.handled, hook.written, hook.err)
		}
	} else if hook.called {
		// poll.Splice will certainly not be called when the source
		// is a wrapper and the destination is not TCPConn.
		t.Errorf("expected poll.Splice not be called")
	}
}

func verifySpliceFds(t *testing.T, c Conn, hook *spliceHook, fdType string) {
	t.Helper()

	sc, ok := c.(syscall.Conn)
	if !ok {
		t.Fatalf("expected syscall.Conn")
	}
	rc, err := sc.SyscallConn()
	if err != nil {
		t.Fatalf("syscall.Conn.SyscallConn error: %v", err)
	}
	var hookFd int
	switch fdType {
	case "src":
		hookFd = hook.srcfd
	case "dst":
		hookFd = hook.dstfd
	default:
		t.Fatalf("unknown fdType %q", fdType)
	}
	if err := rc.Control(func(fd uintptr) {
		if hook.called && hookFd != int(fd) {
			t.Fatalf("wrong %s file descriptor: got %d, want %d", fdType, hook.dstfd, int(fd))
		}
	}); err != nil {
		t.Fatalf("syscall.RawConn.Control error: %v", err)
	}
}

func (tc spliceTestCase) testFile(t *testing.T) {
	hook := hookSplice(t)

	// We need to use the actual size for startTestSocketPeer when testing with LimitedReader,
	// otherwise the child process created in startTestSocketPeer will hang infinitely because of
	// the mismatch of data size to transfer.
	actualSize := tc.totalSize
	if tc.limitReadSize > 0 {
		if tc.limitReadSize < actualSize {
			actualSize = tc.limitReadSize
		}
	}

	f, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	client, server := spawnTestSocketPair(t, tc.upNet)
	defer server.Close()

	cleanup, err := startTestSocketPeer(t, client, "w", tc.chunkSize, actualSize)
	if err != nil {
		client.Close()
		t.Fatal("failed to start splice client:", err)
	}
	defer cleanup(t)

	var r io.Reader = server
	if tc.limitReadSize > 0 {
		r = &io.LimitedReader{
			N: int64(tc.limitReadSize),
			R: r,
		}
	}

	got, err := io.Copy(f, r)
	if err != nil {
		t.Fatalf("failed to ReadFrom with error: %v", err)
	}

	// We shouldn't have called poll.Splice in TCPConn.WriteTo,
	// it's supposed to be called from File.ReadFrom.
	if got > 0 && hook.called {
		t.Error("expected not poll.Splice to be called")
	}

	if want := int64(actualSize); got != want {
		t.Errorf("got %d bytes, want %d", got, want)
	}
	if tc.limitReadSize > 0 {
		wantN := 0
		if tc.limitReadSize > actualSize {
			wantN = tc.limitReadSize - actualSize
		}

		if gotN := r.(*io.LimitedReader).N; gotN != int64(wantN) {
			t.Errorf("r.N = %d, want %d", gotN, wantN)
		}
	}
}

func testSpliceReaderAtEOF(t *testing.T, upNet, downNet string) {
	// UnixConn doesn't implement io.ReaderFrom, which will fail
	// the following test in asserting a UnixConn to be an io.ReaderFrom,
	// so skip this test.
	if downNet == "unix" {
		t.Skip("skipping test on unix socket")
	}

	hook := hookSplice(t)

	clientUp, serverUp := spawnTestSocketPair(t, upNet)
	defer clientUp.Close()
	clientDown, serverDown := spawnTestSocketPair(t, downNet)
	defer clientDown.Close()
	defer serverDown.Close()

	serverUp.Close()

	// We'd like to call net.spliceFrom here and check the handled return
	// value, but we disable splice on old Linux kernels.
	//
	// In that case, poll.Splice and net.spliceFrom return a non-nil error
	// and handled == false. We'd ideally like to see handled == true
	// because the source reader is at EOF, but if we're running on an old
	// kernel, and splice is disabled, we won't see EOF from net.spliceFrom,
	// because we won't touch the reader at all.
	//
	// Trying to untangle the errors from net.spliceFrom and match them
	// against the errors created by the poll package would be brittle,
	// so this is a higher level test.
	//
	// The following ReadFrom should return immediately, regardless of
	// whether splice is disabled or not. The other side should then
	// get a goodbye signal. Test for the goodbye signal.
	msg := "bye"
	go func() {
		serverDown.(io.ReaderFrom).ReadFrom(serverUp)
		io.WriteString(serverDown, msg)
	}()

	buf := make([]byte, 3)
	n, err := io.ReadFull(clientDown, buf)
	if err != nil {
		t.Errorf("clientDown: %v", err)
	}
	if string(buf) != msg {
		t.Errorf("clientDown got %q, want %q", buf, msg)
	}

	// We should have called poll.Splice with the right file descriptor arguments.
	if n > 0 && !hook.called {
		t.Fatal("expected poll.Splice to be called")
	}

	verifySpliceFds(t, serverDown, hook, "dst")

	// poll.Splice is expected to handle the data transmission but fail
	// when working with a closed endpoint, return an error.
	if !hook.handled || hook.written > 0 || hook.err == nil {
		t.Errorf("expected handled = true, written = 0, err != nil, but got handled = %t, written = %d, err = %v",
			hook.handled, hook.written, hook.err)
	}
}

func testSpliceIssue25985(t *testing.T, upNet, downNet string) {
	front := newLocalListener(t, upNet)
	defer front.Close()
	back := newLocalListener(t, downNet)
	defer back.Close()

	var wg sync.WaitGroup
	wg.Add(2)

	proxy := func() {
		src, err := front.Accept()
		if err != nil {
			return
		}
		dst, err := Dial(downNet, back.Addr().String())
		if err != nil {
			return
		}
		defer dst.Close()
		defer src.Close()
		go func() {
			io.Copy(src, dst)
			wg.Done()
		}()
		go func() {
			io.Copy(dst, src)
			wg.Done()
		}()
	}

	go proxy()

	toFront, err := Dial(upNet, front.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	io.WriteString(toFront, "foo")
	toFront.Close()

	fromProxy, err := back.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer fromProxy.Close()

	_, err = io.ReadAll(fromProxy)
	if err != nil {
		t.Fatal(err)
	}

	wg.Wait()
}

func testSpliceNoUnixpacket(t *testing.T) {
	clientUp, serverUp := spawnTestSocketPair(t, "unixpacket")
	defer clientUp.Close()
	defer serverUp.Close()
	clientDown, serverDown := spawnTestSocketPair(t, "tcp")
	defer clientDown.Close()
	defer serverDown.Close()
	// If splice called poll.Splice here, we'd get err == syscall.EINVAL
	// and handled == false.  If poll.Splice gets an EINVAL on the first
	// try, it assumes the kernel it's running on doesn't support splice
	// for unix sockets and returns handled == false. This works for our
	// purposes by somewhat of an accident, but is not entirely correct.
	//
	// What we want is err == nil and handled == false, i.e. we never
	// called poll.Splice, because we know the unix socket's network.
	_, err, handled := spliceFrom(serverDown.(*TCPConn).fd, serverUp)
	if err != nil || handled != false {
		t.Fatalf("got err = %v, handled = %t, want nil error, handled == false", err, handled)
	}
}

func testSpliceNoUnixgram(t *testing.T) {
	addr, err := ResolveUnixAddr("unixgram", testUnixAddr(t))
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(addr.Name)
	up, err := ListenUnixgram("unixgram", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer up.Close()
	clientDown, serverDown := spawnTestSocketPair(t, "tcp")
	defer clientDown.Close()
	defer serverDown.Close()
	// Analogous to testSpliceNoUnixpacket.
	_, err, handled := spliceFrom(serverDown.(*TCPConn).fd, up)
	if err != nil || handled != false {
		t.Fatalf("got err = %v, handled = %t, want nil error, handled == false", err, handled)
	}
}

func BenchmarkSplice(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	b.Run("tcp-to-tcp", func(b *testing.B) { benchSplice(b, "tcp", "tcp") })
	b.Run("unix-to-tcp", func(b *testing.B) { benchSplice(b, "unix", "tcp") })
	b.Run("tcp-to-unix", func(b *testing.B) { benchSplice(b, "tcp", "unix") })
}

func benchSplice(b *testing.B, upNet, downNet string) {
	for i := 0; i <= 10; i++ {
		chunkSize := 1 << uint(i+10)
		tc := spliceTestCase{
			upNet:     upNet,
			downNet:   downNet,
			chunkSize: chunkSize,
		}

		b.Run(strconv.Itoa(chunkSize), tc.bench)
	}
}

func (tc spliceTestCase) bench(b *testing.B) {
	// To benchmark the genericReadFrom code path, set this to false.
	useSplice := true

	clientUp, serverUp := spawnTestSocketPair(b, tc.upNet)
	defer serverUp.Close()

	cleanup, err := startTestSocketPeer(b, clientUp, "w", tc.chunkSize, tc.chunkSize*b.N)
	if err != nil {
		b.Fatal(err)
	}
	defer cleanup(b)

	clientDown, serverDown := spawnTestSocketPair(b, tc.downNet)
	defer serverDown.Close()

	cleanup, err = startTestSocketPeer(b, clientDown, "r", tc.chunkSize, tc.chunkSize*b.N)
	if err != nil {
		b.Fatal(err)
	}
	defer cleanup(b)

	b.SetBytes(int64(tc.chunkSize))
	b.ResetTimer()

	if useSplice {
		_, err := io.Copy(serverDown, serverUp)
		if err != nil {
			b.Fatal(err)
		}
	} else {
		type onlyReader struct {
			io.Reader
		}
		_, err := io.Copy(serverDown, onlyReader{serverUp})
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSpliceFile(b *testing.B) {
	b.Run("tcp-to-file", func(b *testing.B) { benchmarkSpliceFile(b, "tcp") })
	b.Run("unix-to-file", func(b *testing.B) { benchmarkSpliceFile(b, "unix") })
}

func benchmarkSpliceFile(b *testing.B, proto string) {
	for i := 0; i <= 10; i++ {
		size := 1 << (i + 10)
		bench := spliceFileBench{
			proto:     proto,
			chunkSize: size,
		}
		b.Run(strconv.Itoa(size), bench.benchSpliceFile)
	}
}

type spliceFileBench struct {
	proto     string
	chunkSize int
}

func (bench spliceFileBench) benchSpliceFile(b *testing.B) {
	f, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	totalSize := b.N * bench.chunkSize

	client, server := spawnTestSocketPair(b, bench.proto)
	defer server.Close()

	cleanup, err := startTestSocketPeer(b, client, "w", bench.chunkSize, totalSize)
	if err != nil {
		client.Close()
		b.Fatalf("failed to start splice client: %v", err)
	}
	defer cleanup(b)

	b.ReportAllocs()
	b.SetBytes(int64(bench.chunkSize))
	b.ResetTimer()

	got, err := io.Copy(f, server)
	if err != nil {
		b.Fatalf("failed to ReadFrom with error: %v", err)
	}
	if want := int64(totalSize); got != want {
		b.Errorf("bytes sent mismatch, got: %d, want: %d", got, want)
	}
}

func hookSplice(t *testing.T) *spliceHook {
	t.Helper()

	h := new(spliceHook)
	h.install()
	t.Cleanup(h.uninstall)
	return h
}

type spliceHook struct {
	called bool
	dstfd  int
	srcfd  int
	remain int64

	written int64
	handled bool
	err     error

	original func(dst, src *poll.FD, remain int64) (int64, bool, error)
}

func (h *spliceHook) install() {
	h.original = pollSplice
	pollSplice = func(dst, src *poll.FD, remain int64) (int64, bool, error) {
		h.called = true
		h.dstfd = dst.Sysfd
		h.srcfd = src.Sysfd
		h.remain = remain
		h.written, h.handled, h.err = h.original(dst, src, remain)
		return h.written, h.handled, h.err
	}
}

func (h *spliceHook) uninstall() {
	pollSplice = h.original
}
