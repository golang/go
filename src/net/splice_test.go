// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package net

import (
	"io"
	"log"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"testing"
	"time"
)

func TestSplice(t *testing.T) {
	t.Run("tcp-to-tcp", func(t *testing.T) { testSplice(t, "tcp", "tcp") })
	if !testableNetwork("unixgram") {
		t.Skip("skipping unix-to-tcp tests")
	}
	t.Run("unix-to-tcp", func(t *testing.T) { testSplice(t, "unix", "tcp") })
	t.Run("no-unixpacket", testSpliceNoUnixpacket)
	t.Run("no-unixgram", testSpliceNoUnixgram)
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
	clientUp, serverUp, err := spliceTestSocketPair(tc.upNet)
	if err != nil {
		t.Fatal(err)
	}
	defer serverUp.Close()
	cleanup, err := startSpliceClient(clientUp, "w", tc.chunkSize, tc.totalSize)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	clientDown, serverDown, err := spliceTestSocketPair(tc.downNet)
	if err != nil {
		t.Fatal(err)
	}
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
	serverDown.Close()
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
}

func testSpliceReaderAtEOF(t *testing.T, upNet, downNet string) {
	clientUp, serverUp, err := spliceTestSocketPair(upNet)
	if err != nil {
		t.Fatal(err)
	}
	defer clientUp.Close()
	clientDown, serverDown, err := spliceTestSocketPair(downNet)
	if err != nil {
		t.Fatal(err)
	}
	defer clientDown.Close()

	serverUp.Close()

	// We'd like to call net.splice here and check the handled return
	// value, but we disable splice on old Linux kernels.
	//
	// In that case, poll.Splice and net.splice return a non-nil error
	// and handled == false. We'd ideally like to see handled == true
	// because the source reader is at EOF, but if we're running on an old
	// kernel, and splice is disabled, we won't see EOF from net.splice,
	// because we won't touch the reader at all.
	//
	// Trying to untangle the errors from net.splice and match them
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
		serverDown.Close()
	}()

	buf := make([]byte, 3)
	_, err = io.ReadFull(clientDown, buf)
	if err != nil {
		t.Errorf("clientDown: %v", err)
	}
	if string(buf) != msg {
		t.Errorf("clientDown got %q, want %q", buf, msg)
	}
}

func testSpliceIssue25985(t *testing.T, upNet, downNet string) {
	front, err := newLocalListener(upNet)
	if err != nil {
		t.Fatal(err)
	}
	defer front.Close()
	back, err := newLocalListener(downNet)
	if err != nil {
		t.Fatal(err)
	}
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
	clientUp, serverUp, err := spliceTestSocketPair("unixpacket")
	if err != nil {
		t.Fatal(err)
	}
	defer clientUp.Close()
	defer serverUp.Close()
	clientDown, serverDown, err := spliceTestSocketPair("tcp")
	if err != nil {
		t.Fatal(err)
	}
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
	_, err, handled := splice(serverDown.(*TCPConn).fd, serverUp)
	if err != nil || handled != false {
		t.Fatalf("got err = %v, handled = %t, want nil error, handled == false", err, handled)
	}
}

func testSpliceNoUnixgram(t *testing.T) {
	addr, err := ResolveUnixAddr("unixgram", testUnixAddr())
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(addr.Name)
	up, err := ListenUnixgram("unixgram", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer up.Close()
	clientDown, serverDown, err := spliceTestSocketPair("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer clientDown.Close()
	defer serverDown.Close()
	// Analogous to testSpliceNoUnixpacket.
	_, err, handled := splice(serverDown.(*TCPConn).fd, up)
	if err != nil || handled != false {
		t.Fatalf("got err = %v, handled = %t, want nil error, handled == false", err, handled)
	}
}

func BenchmarkSplice(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	b.Run("tcp-to-tcp", func(b *testing.B) { benchSplice(b, "tcp", "tcp") })
	b.Run("unix-to-tcp", func(b *testing.B) { benchSplice(b, "unix", "tcp") })
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

	clientUp, serverUp, err := spliceTestSocketPair(tc.upNet)
	if err != nil {
		b.Fatal(err)
	}
	defer serverUp.Close()

	cleanup, err := startSpliceClient(clientUp, "w", tc.chunkSize, tc.chunkSize*b.N)
	if err != nil {
		b.Fatal(err)
	}
	defer cleanup()

	clientDown, serverDown, err := spliceTestSocketPair(tc.downNet)
	if err != nil {
		b.Fatal(err)
	}
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

func spliceTestSocketPair(net string) (client, server Conn, err error) {
	ln, err := newLocalListener(net)
	if err != nil {
		return nil, nil, err
	}
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
		return nil, nil, cerr
	}
	if serr != nil {
		if client != nil {
			client.Close()
		}
		return nil, nil, serr
	}
	return client, server, nil
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
