// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package net

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"sync"
	"testing"
)

func TestSplice(t *testing.T) {
	t.Run("simple", testSpliceSimple)
	t.Run("multipleWrite", testSpliceMultipleWrite)
	t.Run("big", testSpliceBig)
	t.Run("honorsLimitedReader", testSpliceHonorsLimitedReader)
	t.Run("readerAtEOF", testSpliceReaderAtEOF)
	t.Run("issue25985", testSpliceIssue25985)
}

func testSpliceSimple(t *testing.T) {
	srv, err := newSpliceTestServer()
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	copyDone := srv.Copy()
	msg := []byte("splice test")
	if _, err := srv.Write(msg); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, len(msg))
	if _, err := io.ReadFull(srv, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, msg) {
		t.Errorf("got %q, wrote %q", got, msg)
	}
	srv.CloseWrite()
	srv.CloseRead()
	if err := <-copyDone; err != nil {
		t.Errorf("splice: %v", err)
	}
}

func testSpliceMultipleWrite(t *testing.T) {
	srv, err := newSpliceTestServer()
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	copyDone := srv.Copy()
	msg1 := []byte("splice test part 1 ")
	msg2 := []byte(" splice test part 2")
	if _, err := srv.Write(msg1); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if _, err := srv.Write(msg2); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, len(msg1)+len(msg2))
	if _, err := io.ReadFull(srv, got); err != nil {
		t.Fatal(err)
	}
	want := append(msg1, msg2...)
	if !bytes.Equal(got, want) {
		t.Errorf("got %q, wrote %q", got, want)
	}
	srv.CloseWrite()
	srv.CloseRead()
	if err := <-copyDone; err != nil {
		t.Errorf("splice: %v", err)
	}
}

func testSpliceBig(t *testing.T) {
	// The maximum amount of data that internal/poll.Splice will use in a
	// splice(2) call is 4 << 20. Use a bigger size here so that we test an
	// amount that doesn't fit in a single call.
	size := 5 << 20
	srv, err := newSpliceTestServer()
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	big := make([]byte, size)
	copyDone := srv.Copy()
	type readResult struct {
		b   []byte
		err error
	}
	readDone := make(chan readResult)
	go func() {
		got := make([]byte, len(big))
		_, err := io.ReadFull(srv, got)
		readDone <- readResult{got, err}
	}()
	if _, err := srv.Write(big); err != nil {
		t.Fatal(err)
	}
	res := <-readDone
	if res.err != nil {
		t.Fatal(res.err)
	}
	got := res.b
	if !bytes.Equal(got, big) {
		t.Errorf("input and output differ")
	}
	srv.CloseWrite()
	srv.CloseRead()
	if err := <-copyDone; err != nil {
		t.Errorf("splice: %v", err)
	}
}

func testSpliceHonorsLimitedReader(t *testing.T) {
	t.Run("stopsAfterN", testSpliceStopsAfterN)
	t.Run("updatesN", testSpliceUpdatesN)
	t.Run("readerAtLimit", testSpliceReaderAtLimit)
}

func testSpliceStopsAfterN(t *testing.T) {
	clientUp, serverUp, err := spliceTestSocketPair("tcp")
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
	count := 128
	copyDone := make(chan error)
	lr := &io.LimitedReader{
		N: int64(count),
		R: serverUp,
	}
	go func() {
		_, err := io.Copy(serverDown, lr)
		serverDown.Close()
		copyDone <- err
	}()
	msg := make([]byte, 2*count)
	if _, err := clientUp.Write(msg); err != nil {
		t.Fatal(err)
	}
	clientUp.Close()
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, clientDown); err != nil {
		t.Fatal(err)
	}
	if buf.Len() != count {
		t.Errorf("splice transferred %d bytes, want to stop after %d", buf.Len(), count)
	}
	clientDown.Close()
	if err := <-copyDone; err != nil {
		t.Errorf("splice: %v", err)
	}
}

func testSpliceUpdatesN(t *testing.T) {
	clientUp, serverUp, err := spliceTestSocketPair("tcp")
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
	count := 128
	copyDone := make(chan error)
	lr := &io.LimitedReader{
		N: int64(100 + count),
		R: serverUp,
	}
	go func() {
		_, err := io.Copy(serverDown, lr)
		copyDone <- err
	}()
	msg := make([]byte, count)
	if _, err := clientUp.Write(msg); err != nil {
		t.Fatal(err)
	}
	clientUp.Close()
	got := make([]byte, count)
	if _, err := io.ReadFull(clientDown, got); err != nil {
		t.Fatal(err)
	}
	clientDown.Close()
	if err := <-copyDone; err != nil {
		t.Errorf("splice: %v", err)
	}
	wantN := int64(100)
	if lr.N != wantN {
		t.Errorf("lr.N = %d, want %d", lr.N, wantN)
	}
}

func testSpliceReaderAtLimit(t *testing.T) {
	clientUp, serverUp, err := spliceTestSocketPair("tcp")
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

	lr := &io.LimitedReader{
		N: 0,
		R: serverUp,
	}
	_, err, handled := splice(serverDown.(*TCPConn).fd, lr)
	if !handled {
		t.Errorf("exhausted LimitedReader: got err = %v, handled = %t, want handled = true", err, handled)
	}
}

func testSpliceReaderAtEOF(t *testing.T) {
	clientUp, serverUp, err := spliceTestSocketPair("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer clientUp.Close()
	clientDown, serverDown, err := spliceTestSocketPair("tcp")
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
		serverDown.(*TCPConn).ReadFrom(serverUp)
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

func testSpliceIssue25985(t *testing.T) {
	front, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer front.Close()
	back, err := newLocalListener("tcp")
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
		dst, err := Dial("tcp", back.Addr().String())
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

	toFront, err := Dial("tcp", front.Addr().String())
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

	_, err = ioutil.ReadAll(fromProxy)
	if err != nil {
		t.Fatal(err)
	}

	wg.Wait()
}

func BenchmarkTCPReadFrom(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	var chunkSizes []int
	for i := uint(10); i <= 20; i++ {
		chunkSizes = append(chunkSizes, 1<<i)
	}
	// To benchmark the genericReadFrom code path, set this to false.
	useSplice := true
	for _, chunkSize := range chunkSizes {
		b.Run(fmt.Sprint(chunkSize), func(b *testing.B) {
			benchmarkSplice(b, chunkSize, useSplice)
		})
	}
}

func benchmarkSplice(b *testing.B, chunkSize int, useSplice bool) {
	srv, err := newSpliceTestServer()
	if err != nil {
		b.Fatal(err)
	}
	defer srv.Close()
	var copyDone <-chan error
	if useSplice {
		copyDone = srv.Copy()
	} else {
		copyDone = srv.CopyNoSplice()
	}
	chunk := make([]byte, chunkSize)
	discardDone := make(chan struct{})
	go func() {
		for {
			buf := make([]byte, chunkSize)
			_, err := srv.Read(buf)
			if err != nil {
				break
			}
		}
		discardDone <- struct{}{}
	}()
	b.SetBytes(int64(chunkSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srv.Write(chunk)
	}
	srv.CloseWrite()
	<-copyDone
	srv.CloseRead()
	<-discardDone
}

type spliceTestServer struct {
	clientUp   io.WriteCloser
	clientDown io.ReadCloser
	serverUp   io.ReadCloser
	serverDown io.WriteCloser
}

func newSpliceTestServer() (*spliceTestServer, error) {
	// For now, both networks are hard-coded to TCP.
	// If splice is enabled for non-tcp upstream connections,
	// newSpliceTestServer will need to take a network parameter.
	clientUp, serverUp, err := spliceTestSocketPair("tcp")
	if err != nil {
		return nil, err
	}
	clientDown, serverDown, err := spliceTestSocketPair("tcp")
	if err != nil {
		clientUp.Close()
		serverUp.Close()
		return nil, err
	}
	return &spliceTestServer{clientUp, clientDown, serverUp, serverDown}, nil
}

// Read reads from the downstream connection.
func (srv *spliceTestServer) Read(b []byte) (int, error) {
	return srv.clientDown.Read(b)
}

// Write writes to the upstream connection.
func (srv *spliceTestServer) Write(b []byte) (int, error) {
	return srv.clientUp.Write(b)
}

// Close closes the server.
func (srv *spliceTestServer) Close() error {
	err := srv.closeUp()
	err1 := srv.closeDown()
	if err == nil {
		return err1
	}
	return err
}

// CloseWrite closes the client side of the upstream connection.
func (srv *spliceTestServer) CloseWrite() error {
	return srv.clientUp.Close()
}

// CloseRead closes the client side of the downstream connection.
func (srv *spliceTestServer) CloseRead() error {
	return srv.clientDown.Close()
}

// Copy copies from the server side of the upstream connection
// to the server side of the downstream connection, in a separate
// goroutine. Copy is done when the first send on the returned
// channel succeeds.
func (srv *spliceTestServer) Copy() <-chan error {
	ch := make(chan error)
	go func() {
		_, err := io.Copy(srv.serverDown, srv.serverUp)
		ch <- err
		close(ch)
	}()
	return ch
}

// CopyNoSplice is like Copy, but ensures that the splice code path
// is not reached.
func (srv *spliceTestServer) CopyNoSplice() <-chan error {
	type onlyReader struct {
		io.Reader
	}
	ch := make(chan error)
	go func() {
		_, err := io.Copy(srv.serverDown, onlyReader{srv.serverUp})
		ch <- err
		close(ch)
	}()
	return ch
}

func (srv *spliceTestServer) closeUp() error {
	var err, err1 error
	if srv.serverUp != nil {
		err = srv.serverUp.Close()
	}
	if srv.clientUp != nil {
		err1 = srv.clientUp.Close()
	}
	if err == nil {
		return err1
	}
	return err
}

func (srv *spliceTestServer) closeDown() error {
	var err, err1 error
	if srv.serverDown != nil {
		err = srv.serverDown.Close()
	}
	if srv.clientDown != nil {
		err1 = srv.clientDown.Close()
	}
	if err == nil {
		return err1
	}
	return err
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
