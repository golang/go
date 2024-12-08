// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"internal/poll"
	"io"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"
)

const (
	newton       = "../testdata/Isaac.Newton-Opticks.txt"
	newtonLen    = 567198
	newtonSHA256 = "d4a9ac22462b35e7821a4f2706c211093da678620a8f9997989ee7cf8d507bbd"
)

// expectSendfile runs f, and verifies that internal/poll.SendFile successfully handles
// a write to wantConn during f's execution.
//
// On platforms where supportsSendfile is false, expectSendfile runs f but does not
// expect a call to SendFile.
func expectSendfile(t *testing.T, wantConn Conn, f func()) {
	t.Helper()
	if !supportsSendfile {
		f()
		return
	}
	orig := poll.TestHookDidSendFile
	defer func() {
		poll.TestHookDidSendFile = orig
	}()
	var (
		called     bool
		gotHandled bool
		gotFD      *poll.FD
		gotErr     error
	)
	poll.TestHookDidSendFile = func(dstFD *poll.FD, src int, written int64, err error, handled bool) {
		if called {
			t.Error("internal/poll.SendFile called multiple times, want one call")
		}
		called = true
		gotHandled = handled
		gotFD = dstFD
		gotErr = err
	}
	f()
	if !called {
		t.Error("internal/poll.SendFile was not called, want it to be")
		return
	}
	if !gotHandled {
		t.Error("internal/poll.SendFile did not handle the write, want it to, error:", gotErr)
		return
	}
	if &wantConn.(*TCPConn).fd.pfd != gotFD {
		t.Error("internal.poll.SendFile called with unexpected FD")
	}
}

func TestSendfile(t *testing.T) { testSendfile(t, newton, newtonSHA256, newtonLen, 0) }
func TestSendfileWithExactLimit(t *testing.T) {
	testSendfile(t, newton, newtonSHA256, newtonLen, newtonLen)
}
func TestSendfileWithLimitLargerThanFile(t *testing.T) {
	testSendfile(t, newton, newtonSHA256, newtonLen, newtonLen*2)
}
func TestSendfileWithLargeFile(t *testing.T) {
	// Some platforms are not capable of handling large files with sendfile
	// due to limited system resource, so we only run this test on amd64 and
	// arm64 for the moment.
	if runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64" {
		t.Skip("skipping on non-amd64 and non-arm64 platforms")
	}
	// Also skip it during short testing.
	if testing.Short() {
		t.Skip("Skip it during short testing")
	}

	// We're using 1<<31 - 1 as the chunk size for sendfile currently,
	// make an edge case file that is 1 byte bigger than that.
	f := createTempFile(t, 1<<31)
	// For big file like this, only verify the transmission of the file,
	// skip the content check.
	testSendfile(t, f.Name(), "", 1<<31, 0)
}
func testSendfile(t *testing.T, filePath, fileHash string, size, limit int64) {
	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	errc := make(chan error, 1)
	go func(ln Listener) {
		// Wait for a connection.
		conn, err := ln.Accept()
		if err != nil {
			errc <- err
			close(errc)
			return
		}

		go func() {
			defer close(errc)
			defer conn.Close()

			f, err := os.Open(filePath)
			if err != nil {
				errc <- err
				return
			}
			defer f.Close()

			// Return file data using io.Copy, which should use
			// sendFile if available.
			var sbytes int64
			switch runtime.GOOS {
			case "windows":
				// Windows is not using sendfile for some reason:
				// https://go.dev/issue/67042
				sbytes, err = io.Copy(conn, f)
			default:
				expectSendfile(t, conn, func() {
					if limit > 0 {
						sbytes, err = io.CopyN(conn, f, limit)
						if err == io.EOF && limit > size {
							err = nil
						}
					} else {
						sbytes, err = io.Copy(conn, f)
					}
				})
			}
			if err != nil {
				errc <- err
				return
			}

			if sbytes != size {
				errc <- fmt.Errorf("sent %d bytes; expected %d", sbytes, size)
				return
			}
		}()
	}(ln)

	// Connect to listener to retrieve file and verify digest matches
	// expected.
	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	h := sha256.New()
	rbytes, err := io.Copy(h, c)
	if err != nil {
		t.Error(err)
	}

	if rbytes != size {
		t.Errorf("received %d bytes; expected %d", rbytes, size)
	}

	if len(fileHash) > 0 && hex.EncodeToString(h.Sum(nil)) != newtonSHA256 {
		t.Error("retrieved data hash did not match")
	}

	for err := range errc {
		t.Error(err)
	}
}

func TestSendfileParts(t *testing.T) {
	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	errc := make(chan error, 1)
	go func(ln Listener) {
		// Wait for a connection.
		conn, err := ln.Accept()
		if err != nil {
			errc <- err
			close(errc)
			return
		}

		go func() {
			defer close(errc)
			defer conn.Close()

			f, err := os.Open(newton)
			if err != nil {
				errc <- err
				return
			}
			defer f.Close()

			for i := 0; i < 3; i++ {
				// Return file data using io.CopyN, which should use
				// sendFile if available.
				expectSendfile(t, conn, func() {
					_, err = io.CopyN(conn, f, 3)
				})
				if err != nil {
					errc <- err
					return
				}
			}
		}()
	}(ln)

	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	buf := new(bytes.Buffer)
	buf.ReadFrom(c)

	if want, have := "Produced ", buf.String(); have != want {
		t.Errorf("unexpected server reply %q, want %q", have, want)
	}

	for err := range errc {
		t.Error(err)
	}
}

func TestSendfileSeeked(t *testing.T) {
	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	const seekTo = 65 << 10
	const sendSize = 10 << 10

	errc := make(chan error, 1)
	go func(ln Listener) {
		// Wait for a connection.
		conn, err := ln.Accept()
		if err != nil {
			errc <- err
			close(errc)
			return
		}

		go func() {
			defer close(errc)
			defer conn.Close()

			f, err := os.Open(newton)
			if err != nil {
				errc <- err
				return
			}
			defer f.Close()
			if _, err := f.Seek(seekTo, io.SeekStart); err != nil {
				errc <- err
				return
			}

			expectSendfile(t, conn, func() {
				_, err = io.CopyN(conn, f, sendSize)
			})
			if err != nil {
				errc <- err
				return
			}
		}()
	}(ln)

	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	buf := new(bytes.Buffer)
	buf.ReadFrom(c)

	if buf.Len() != sendSize {
		t.Errorf("Got %d bytes; want %d", buf.Len(), sendSize)
	}

	for err := range errc {
		t.Error(err)
	}
}

// Test that sendfile doesn't put a pipe into blocking mode.
func TestSendfilePipe(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows", "js", "wasip1":
		// These systems don't support deadlines on pipes.
		t.Skipf("skipping on %s", runtime.GOOS)
	}

	t.Parallel()

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	defer r.Close()

	copied := make(chan bool)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		// Accept a connection and copy 1 byte from the read end of
		// the pipe to the connection. This will call into sendfile.
		defer wg.Done()
		conn, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		// The comment above states that this should call into sendfile,
		// but empirically it doesn't seem to do so at this time.
		// If it does, or does on some platforms, this CopyN should be wrapped
		// in expectSendfile.
		_, err = io.CopyN(conn, r, 1)
		if err != nil {
			t.Error(err)
			return
		}
		// Signal the main goroutine that we've copied the byte.
		close(copied)
	}()

	wg.Add(1)
	go func() {
		// Write 1 byte to the write end of the pipe.
		defer wg.Done()
		_, err := w.Write([]byte{'a'})
		if err != nil {
			t.Error(err)
		}
	}()

	wg.Add(1)
	go func() {
		// Connect to the server started two goroutines up and
		// discard any data that it writes.
		defer wg.Done()
		conn, err := Dial("tcp", ln.Addr().String())
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		io.Copy(io.Discard, conn)
	}()

	// Wait for the byte to be copied, meaning that sendfile has
	// been called on the pipe.
	<-copied

	// Set a very short deadline on the read end of the pipe.
	if err := r.SetDeadline(time.Now().Add(time.Microsecond)); err != nil {
		t.Fatal(err)
	}

	wg.Add(1)
	go func() {
		// Wait for much longer than the deadline and write a byte
		// to the pipe.
		defer wg.Done()
		time.Sleep(50 * time.Millisecond)
		w.Write([]byte{'b'})
	}()

	// If this read does not time out, the pipe was incorrectly
	// put into blocking mode.
	_, err = r.Read(make([]byte, 1))
	if err == nil {
		t.Error("Read did not time out")
	} else if !os.IsTimeout(err) {
		t.Errorf("got error %v, expected a time out", err)
	}

	wg.Wait()
}

// Issue 43822: tests that returns EOF when conn write timeout.
func TestSendfileOnWriteTimeoutExceeded(t *testing.T) {
	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	errc := make(chan error, 1)
	go func(ln Listener) (retErr error) {
		defer func() {
			errc <- retErr
			close(errc)
		}()

		conn, err := ln.Accept()
		if err != nil {
			return err
		}
		defer conn.Close()

		// Set the write deadline in the past(1h ago). It makes
		// sure that it is always write timeout.
		if err := conn.SetWriteDeadline(time.Now().Add(-1 * time.Hour)); err != nil {
			return err
		}

		f, err := os.Open(newton)
		if err != nil {
			return err
		}
		defer f.Close()

		// We expect this to use sendfile, but as of the time this comment was written
		// poll.SendFile on an FD past its timeout can return an error indicating that
		// it didn't handle the operation, resulting in a non-sendfile retry.
		// So don't use expectSendfile here.
		_, err = io.Copy(conn, f)
		if errors.Is(err, os.ErrDeadlineExceeded) {
			return nil
		}

		if err == nil {
			err = fmt.Errorf("expected ErrDeadlineExceeded, but got nil")
		}
		return err
	}(ln)

	conn, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	n, err := io.Copy(io.Discard, conn)
	if err != nil {
		t.Fatalf("expected nil error, but got %v", err)
	}
	if n != 0 {
		t.Fatalf("expected receive zero, but got %d byte(s)", n)
	}

	if err := <-errc; err != nil {
		t.Fatal(err)
	}
}

func BenchmarkSendfileZeroBytes(b *testing.B) {
	var (
		wg          sync.WaitGroup
		ctx, cancel = context.WithCancel(context.Background())
	)

	defer wg.Wait()

	ln := newLocalListener(b, "tcp")
	defer ln.Close()

	tempFile, err := os.CreateTemp(b.TempDir(), "test.txt")
	if err != nil {
		b.Fatalf("failed to create temp file: %v", err)
	}
	defer tempFile.Close()

	fileName := tempFile.Name()

	dataSize := b.N
	wg.Add(1)
	go func(f *os.File) {
		defer wg.Done()

		for i := 0; i < dataSize; i++ {
			if _, err := f.Write([]byte{1}); err != nil {
				b.Errorf("failed to write: %v", err)
				return
			}
			if i%1000 == 0 {
				f.Sync()
			}
		}
	}(tempFile)

	b.ResetTimer()
	b.ReportAllocs()

	wg.Add(1)
	go func(ln Listener, fileName string) {
		defer wg.Done()

		conn, err := ln.Accept()
		if err != nil {
			b.Errorf("failed to accept: %v", err)
			return
		}
		defer conn.Close()

		f, err := os.OpenFile(fileName, os.O_RDONLY, 0660)
		if err != nil {
			b.Errorf("failed to open file: %v", err)
			return
		}
		defer f.Close()

		for {
			if ctx.Err() != nil {
				return
			}

			if _, err := io.Copy(conn, f); err != nil {
				b.Errorf("failed to copy: %v", err)
				return
			}
		}
	}(ln, fileName)

	conn, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		b.Fatalf("failed to dial: %v", err)
	}
	defer conn.Close()

	n, err := io.CopyN(io.Discard, conn, int64(dataSize))
	if err != nil {
		b.Fatalf("failed to copy: %v", err)
	}
	if n != int64(dataSize) {
		b.Fatalf("expected %d copied bytes, but got %d", dataSize, n)
	}

	cancel()
}

func BenchmarkSendFile(b *testing.B) {
	if runtime.GOOS == "windows" {
		// TODO(panjf2000): Windows has not yet implemented FileConn,
		//		remove this when it's implemented in https://go.dev/issues/9503.
		b.Skipf("skipping on %s", runtime.GOOS)
	}

	b.Run("file-to-tcp", func(b *testing.B) { benchmarkSendFile(b, "tcp") })
	b.Run("file-to-unix", func(b *testing.B) { benchmarkSendFile(b, "unix") })
}

func benchmarkSendFile(b *testing.B, proto string) {
	for i := 0; i <= 10; i++ {
		size := 1 << (i + 10)
		bench := sendFileBench{
			proto:     proto,
			chunkSize: size,
		}
		b.Run(strconv.Itoa(size), bench.benchSendFile)
	}
}

type sendFileBench struct {
	proto     string
	chunkSize int
}

func (bench sendFileBench) benchSendFile(b *testing.B) {
	fileSize := b.N * bench.chunkSize
	f := createTempFile(b, int64(fileSize))

	client, server := spawnTestSocketPair(b, bench.proto)
	defer server.Close()

	cleanUp, err := startTestSocketPeer(b, client, "r", bench.chunkSize, fileSize)
	if err != nil {
		client.Close()
		b.Fatal(err)
	}
	defer cleanUp(b)

	b.ReportAllocs()
	b.SetBytes(int64(bench.chunkSize))
	b.ResetTimer()

	// Data go from file to socket via sendfile(2).
	sent, err := io.Copy(server, f)
	if err != nil {
		b.Fatalf("failed to copy data with sendfile, error: %v", err)
	}
	if sent != int64(fileSize) {
		b.Fatalf("bytes sent mismatch, got: %d, want: %d", sent, fileSize)
	}
}

func createTempFile(tb testing.TB, size int64) *os.File {
	f, err := os.CreateTemp(tb.TempDir(), "sendfile-bench")
	if err != nil {
		tb.Fatalf("failed to create temporary file: %v", err)
	}
	tb.Cleanup(func() {
		f.Close()
	})

	if _, err := io.CopyN(f, newRandReader(tb), size); err != nil {
		tb.Fatalf("failed to fill the file with random data: %v", err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		tb.Fatalf("failed to rewind the file: %v", err)
	}

	return f
}

func newRandReader(tb testing.TB) io.Reader {
	seed := time.Now().UnixNano()
	tb.Logf("Deterministic RNG seed based on timestamp: 0x%x", seed)
	return rand.New(rand.NewSource(seed))
}
