// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"os"
	"runtime"
	"sync"
	"testing"

	"golang.org/x/net/nettest"
)

// Exercise sendfile/splice fast paths with a moderately large file.
//
// https://go.dev/issue/70000

func TestLargeCopyViaNetwork(t *testing.T) {
	const size = 10 * 1024 * 1024
	dir := t.TempDir()

	src, err := os.Create(dir + "/src")
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if _, err := io.CopyN(src, newRandReader(), size); err != nil {
		t.Fatal(err)
	}
	if _, err := src.Seek(0, 0); err != nil {
		t.Fatal(err)
	}

	dst, err := os.Create(dir + "/dst")
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()

	client, server := createSocketPair(t, "tcp")
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		if n, err := io.Copy(dst, server); n != size || err != nil {
			t.Errorf("copy to destination = %v, %v; want %v, nil", n, err, size)
		}
	}()
	go func() {
		defer wg.Done()
		defer client.Close()
		if n, err := io.Copy(client, src); n != size || err != nil {
			t.Errorf("copy from source = %v, %v; want %v, nil", n, err, size)
		}
	}()
	wg.Wait()

	if _, err := dst.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	if err := compareReaders(dst, io.LimitReader(newRandReader(), size)); err != nil {
		t.Fatal(err)
	}
}

func TestCopyFileToFile(t *testing.T) {
	const size = 1 * 1024 * 1024
	dir := t.TempDir()

	src, err := os.Create(dir + "/src")
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if _, err := io.CopyN(src, newRandReader(), size); err != nil {
		t.Fatal(err)
	}
	if _, err := src.Seek(0, 0); err != nil {
		t.Fatal(err)
	}

	mustSeek := func(f *os.File, offset int64, whence int) int64 {
		ret, err := f.Seek(offset, whence)
		if err != nil {
			t.Fatal(err)
		}
		return ret
	}

	for _, srcStart := range []int64{0, 100, size} {
		remaining := size - srcStart
		for _, dstStart := range []int64{0, 200} {
			for _, limit := range []int64{remaining, remaining - 100, size * 2} {
				if limit < 0 {
					continue
				}
				name := fmt.Sprintf("srcStart=%v/dstStart=%v/limit=%v", srcStart, dstStart, limit)
				t.Run(name, func(t *testing.T) {
					dst, err := os.CreateTemp(dir, "dst")
					if err != nil {
						t.Fatal(err)
					}
					defer dst.Close()
					defer os.Remove(dst.Name())

					mustSeek(src, srcStart, io.SeekStart)
					if _, err := io.CopyN(dst, zeroReader{}, dstStart); err != nil {
						t.Fatal(err)
					}

					var copied int64
					if limit == 0 {
						copied, err = io.Copy(dst, src)
					} else {
						copied, err = io.CopyN(dst, src, limit)
					}
					if limit > remaining {
						if err != io.EOF {
							t.Errorf("Copy: %v; want io.EOF", err)
						}
					} else {
						if err != nil {
							t.Errorf("Copy: %v; want nil", err)
						}
					}

					wantCopied := remaining
					if limit != 0 {
						wantCopied = min(limit, wantCopied)
					}
					if copied != wantCopied {
						t.Errorf("copied %v bytes, want %v", copied, wantCopied)
					}

					srcPos := mustSeek(src, 0, io.SeekCurrent)
					wantSrcPos := srcStart + wantCopied
					if srcPos != wantSrcPos {
						t.Errorf("source position = %v, want %v", srcPos, wantSrcPos)
					}

					dstPos := mustSeek(dst, 0, io.SeekCurrent)
					wantDstPos := dstStart + wantCopied
					if dstPos != wantDstPos {
						t.Errorf("destination position = %v, want %v", dstPos, wantDstPos)
					}

					mustSeek(dst, 0, io.SeekStart)
					rr := newRandReader()
					io.CopyN(io.Discard, rr, srcStart)
					wantReader := io.MultiReader(
						io.LimitReader(zeroReader{}, dstStart),
						io.LimitReader(rr, wantCopied),
					)
					if err := compareReaders(dst, wantReader); err != nil {
						t.Fatal(err)
					}
				})

			}
		}
	}
}

func compareReaders(a, b io.Reader) error {
	bufa := make([]byte, 4096)
	bufb := make([]byte, 4096)
	off := 0
	for {
		na, erra := io.ReadFull(a, bufa)
		if erra != nil && erra != io.EOF && erra != io.ErrUnexpectedEOF {
			return erra
		}
		nb, errb := io.ReadFull(b, bufb)
		if errb != nil && errb != io.EOF && errb != io.ErrUnexpectedEOF {
			return errb
		}
		if !bytes.Equal(bufa[:na], bufb[:nb]) {
			return errors.New("contents mismatch")
		}
		if erra != nil && errb != nil {
			break
		}
		off += len(bufa)
	}
	return nil
}

type zeroReader struct{}

func (r zeroReader) Read(p []byte) (int, error) {
	clear(p)
	return len(p), nil
}

type randReader struct {
	rand *rand.Rand
}

func newRandReader() *randReader {
	return &randReader{rand.New(rand.NewPCG(0, 0))}
}

func (r *randReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(r.rand.Uint32() & 0xff)
	}
	return len(p), nil
}

func createSocketPair(t *testing.T, proto string) (client, server net.Conn) {
	t.Helper()
	if !nettest.TestableNetwork(proto) {
		t.Skipf("%s does not support %q", runtime.GOOS, proto)
	}

	ln, err := nettest.NewLocalListener(proto)
	if err != nil {
		t.Fatalf("NewLocalListener error: %v", err)
	}
	t.Cleanup(func() {
		if ln != nil {
			ln.Close()
		}
		if client != nil {
			client.Close()
		}
		if server != nil {
			server.Close()
		}
	})
	ch := make(chan struct{})
	go func() {
		var err error
		server, err = ln.Accept()
		if err != nil {
			t.Errorf("Accept new connection error: %v", err)
		}
		ch <- struct{}{}
	}()
	client, err = net.Dial(proto, ln.Addr().String())
	<-ch
	if err != nil {
		t.Fatalf("Dial new connection error: %v", err)
	}
	return client, server
}
