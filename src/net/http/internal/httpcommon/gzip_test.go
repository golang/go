// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpcommon

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"io/fs"
	"strings"
	"testing"
	"testing/synctest"
)

// Tests that GzipReader doesn't crash on a second Read call following
// the first Read call's gzip.NewReader returning an error.
func TestGzipReaderDoubleReadCrash(t *testing.T) {
	gz := &GzipReader{Body: io.NopCloser(strings.NewReader("0123456789"))}
	var buf [1]byte
	n, err1 := gz.Read(buf[:])
	if n != 0 || !strings.Contains(fmt.Sprint(err1), "invalid header") {
		t.Fatalf("Read = %v, %v; want 0, invalid header", n, err1)
	}
	n, err2 := gz.Read(buf[:])
	if n != 0 || err2 != err1 {
		t.Fatalf("second Read = %v, %v; want 0, %v", n, err2, err1)
	}
}

func TestGzipReaderReadAfterClose(t *testing.T) {
	var body bytes.Buffer
	w := gzip.NewWriter(&body)
	w.Write([]byte("012345679"))
	w.Close()
	gz := &GzipReader{Body: io.NopCloser(&body)}
	var buf [1]byte
	n, err := gz.Read(buf[:])
	if n != 1 || err != nil {
		t.Fatalf("first Read = %v, %v; want 1, nil", n, err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("gz Close error: %v", err)
	}
	n, err = gz.Read(buf[:])
	if n != 0 || err != fs.ErrClosed {
		t.Fatalf("Read after close = %v, %v; want 0, fs.ErrClosed", n, err)
	}
}

// blockingReadCloser blocks in Read until it is closed.
type blockingReadCloser struct {
	closed chan struct{}
}

func (r *blockingReadCloser) Read([]byte) (int, error) {
	<-r.closed
	return 0, fs.ErrClosed
}

func (r *blockingReadCloser) Close() error {
	close(r.closed)
	return nil
}

// Tests that closing a GzipReader unblocks a Read that is waiting for the
// gzip header, rather than deadlocking on the GzipReader's mutex.
func TestGzipReaderConcurrentCloseAndRead(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		gz := &GzipReader{Body: &blockingReadCloser{closed: make(chan struct{})}}

		readErr := make(chan error, 1)
		go func() {
			var buf [1]byte
			_, err := gz.Read(buf[:])
			readErr <- err
		}()
		synctest.Wait()

		if err := gz.Close(); err != nil {
			t.Fatalf("Close = %v, want nil", err)
		}
		synctest.Wait()

		select {
		case err := <-readErr:
			if err == nil {
				t.Error("Read returned nil error, want error")
			}
		default:
			t.Fatal("Read did not unblock on Close")
		}
	})
}
