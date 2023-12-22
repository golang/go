// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package net

import (
	"internal/poll"
	"io"
	"log"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"testing"
	"time"
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

	clientUp, serverUp := spliceTestSocketPair(t, tc.upNet)
	defer serverUp.Close()
	cleanup, err := startSpliceClient(clientUp, "w", tc.chunkSize, tc.totalSize)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	clientDown, serverDown := spliceTestSocketPair(t, tc.downNet)
	defer serverDown.Close()
	cleanup, err = startSpliceClient(clientDown, "r", tc.chunkSize, tc.totalSize)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	var (
		r    io.Reader = serverUp
		size           = tc.totalSize
	)
	if tc.limitReadSize > 0 {
		if tc.limitReadSize < size {
			size = tc.limitReadSize
		}

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

	f, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	client, server := spliceTestSocketPair(t, tc.upNet)
	defer server.Close()

	cleanup, err := startSpliceClient(client, "w", tc.chunkSize, tc.totalSize)
	if err != nil {
		client.Close()
		t.Fatal("failed to start splice client:", err)
	}
	defer cleanup()

	var (
		r          io.Reader = server
		actualSize           = tc.totalSize
	)
	if tc.limitReadSize > 0 {
		if tc.limitReadSize < actualSize {
			actualSize = tc.limitReadSize
		}

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

	clientUp, serverUp := spliceTestSocketPair(t, upNet)
	defer clientUp.Close()
	clientDown, serverDown := spliceTestSocketPair(t, downNet)
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
	clientUp, serverUp := spliceTestSocketPair(t, "unixpacket")
	defer clientUp.Close()
	defer serverUp.Close()
	clientDown, serverDown := spliceTestSocketPair(t, "tcp")
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
	clientDown, serverDown := spliceTestSocketPair(t, "tcp")
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

	clientUp, serverUp := spliceTestSocketPair(b, tc.upNet)
	defer serverUp.Close()

	cleanup, err := startSpliceClient(clientUp, "w", tc.chunkSize, tc.chunkSize*b.N)
	if err != nil {
		b.Fatal(err)
	}
	defer cleanup()

	clientDown, serverDown := spliceTestSocketPair(b, tc.downNet)
	defer serverDown.Close()

	cleanup, err = startSpliceClient(clientDown, "r", tc.chunkSize, tc.chunkSize*b.N)
	if err != nil {
		b.Fatal(err)
	}
	defer cleanup()

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

func spliceTestSocketPair(t testing.TB, net string) (client, server Conn) {
	t.Helper()
	ln := newLocalListener(t, net)
	defer ln.Close()
	var cerr, serr error
	acceptDone := make(chan struct{})
	go func() {
		server, serr = ln.Accept()
		acceptDone <- struct{}{}
	}()
	client, cerr = Dial(ln.Addr().Network(), ln.Addr().String())
	<-acceptDone
	if cerr != nil {
		if server != nil {
			server.Close()
		}
		t.Fatal(cerr)
	}
	if serr != nil {
		if client != nil {
			client.Close()
		}
		t.Fatal(serr)
	}
	return client, server
}

func startSpliceClient(conn Conn, op string, chunkSize, totalSize int) (func(), error) {
	f, err := conn.(interface{ File() (*os.File, error) }).File()
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(os.Args[0], os.Args[1:]...)
	cmd.Env = []string{
		"GO_NET_TEST_SPLICE=1",
		"GO_NET_TEST_SPLICE_OP=" + op,
		"GO_NET_TEST_SPLICE_CHUNK_SIZE=" + strconv.Itoa(chunkSize),
		"GO_NET_TEST_SPLICE_TOTAL_SIZE=" + strconv.Itoa(totalSize),
		"TMPDIR=" + os.Getenv("TMPDIR"),
	}
	cmd.ExtraFiles = append(cmd.ExtraFiles, f)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	donec := make(chan struct{})
	go func() {
		cmd.Wait()
		conn.Close()
		f.Close()
		close(donec)
	}()

	return func() {
		select {
		case <-donec:
		case <-time.After(5 * time.Second):
			log.Printf("killing splice client after 5 second shutdown timeout")
			cmd.Process.Kill()
			select {
			case <-donec:
			case <-time.After(5 * time.Second):
				log.Printf("splice client didn't die after 10 seconds")
			}
		}
	}, nil
}

func init() {
	if os.Getenv("GO_NET_TEST_SPLICE") == "" {
		return
	}
	defer os.Exit(0)

	f := os.NewFile(uintptr(3), "splice-test-conn")
	defer f.Close()

	conn, err := FileConn(f)
	if err != nil {
		log.Fatal(err)
	}

	var chunkSize int
	if chunkSize, err = strconv.Atoi(os.Getenv("GO_NET_TEST_SPLICE_CHUNK_SIZE")); err != nil {
		log.Fatal(err)
	}
	buf := make([]byte, chunkSize)

	var totalSize int
	if totalSize, err = strconv.Atoi(os.Getenv("GO_NET_TEST_SPLICE_TOTAL_SIZE")); err != nil {
		log.Fatal(err)
	}

	var fn func([]byte) (int, error)
	switch op := os.Getenv("GO_NET_TEST_SPLICE_OP"); op {
	case "r":
		fn = conn.Read
	case "w":
		defer conn.Close()

		fn = conn.Write
	default:
		log.Fatalf("unknown op %q", op)
	}

	var n int
	for count := 0; count < totalSize; count += n {
		if count+chunkSize > totalSize {
			buf = buf[:totalSize-count]
		}

		var err error
		if n, err = fn(buf); err != nil {
			return
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

	client, server := spliceTestSocketPair(b, bench.proto)
	defer server.Close()

	cleanup, err := startSpliceClient(client, "w", bench.chunkSize, totalSize)
	if err != nil {
		client.Close()
		b.Fatalf("failed to start splice client: %v", err)
	}
	defer cleanup()

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
	sc      string
	err     error

	original func(dst, src *poll.FD, remain int64) (int64, bool, string, error)
}

func (h *spliceHook) install() {
	h.original = pollSplice
	pollSplice = func(dst, src *poll.FD, remain int64) (int64, bool, string, error) {
		h.called = true
		h.dstfd = dst.Sysfd
		h.srcfd = src.Sysfd
		h.remain = remain
		h.written, h.handled, h.sc, h.err = h.original(dst, src, remain)
		return h.written, h.handled, h.sc, h.err
	}
}

func (h *spliceHook) uninstall() {
	pollSplice = h.original
}
