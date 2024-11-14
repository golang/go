// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"errors"
	"fmt"
	"io"
	. "net/http"
	"os"
	"sync"
	"testing"
	"time"
)

func TestResponseControllerFlush(t *testing.T) { run(t, testResponseControllerFlush) }
func testResponseControllerFlush(t *testing.T, mode testMode) {
	continuec := make(chan struct{})
	cst := newClientServerTest(t, mode, HandlerFunc(func { w, r ->
		ctl := NewResponseController(w)
		w.Write([]byte("one"))
		if err := ctl.Flush(); err != nil {
			t.Errorf("ctl.Flush() = %v, want nil", err)
			return
		}
		<-continuec
		w.Write([]byte("two"))
	}))

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatalf("unexpected connection error: %v", err)
	}
	defer res.Body.Close()

	buf := make([]byte, 16)
	n, err := res.Body.Read(buf)
	close(continuec)
	if err != nil || string(buf[:n]) != "one" {
		t.Fatalf("Body.Read = %q, %v, want %q, nil", string(buf[:n]), err, "one")
	}

	got, err := io.ReadAll(res.Body)
	if err != nil || string(got) != "two" {
		t.Fatalf("Body.Read = %q, %v, want %q, nil", string(got), err, "two")
	}
}

func TestResponseControllerHijack(t *testing.T) { run(t, testResponseControllerHijack) }
func testResponseControllerHijack(t *testing.T, mode testMode) {
	const header = "X-Header"
	const value = "set"
	cst := newClientServerTest(t, mode, HandlerFunc(func { w, r ->
		ctl := NewResponseController(w)
		c, _, err := ctl.Hijack()
		if mode == http2Mode {
			if err == nil {
				t.Errorf("ctl.Hijack = nil, want error")
			}
			w.Header().Set(header, value)
			return
		}
		if err != nil {
			t.Errorf("ctl.Hijack = _, _, %v, want _, _, nil", err)
			return
		}
		fmt.Fprintf(c, "HTTP/1.0 200 OK\r\n%v: %v\r\nContent-Length: 0\r\n\r\n", header, value)
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := res.Header.Get(header), value; got != want {
		t.Errorf("response header %q = %q, want %q", header, got, want)
	}
}

func TestResponseControllerSetPastWriteDeadline(t *testing.T) {
	run(t, testResponseControllerSetPastWriteDeadline)
}
func testResponseControllerSetPastWriteDeadline(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func { w, r ->
		ctl := NewResponseController(w)
		w.Write([]byte("one"))
		if err := ctl.Flush(); err != nil {
			t.Errorf("before setting deadline: ctl.Flush() = %v, want nil", err)
		}
		if err := ctl.SetWriteDeadline(time.Now().Add(-10 * time.Second)); err != nil {
			t.Errorf("ctl.SetWriteDeadline() = %v, want nil", err)
		}

		w.Write([]byte("two"))
		if err := ctl.Flush(); err == nil {
			t.Errorf("after setting deadline: ctl.Flush() = nil, want non-nil")
		}
		// Connection errors are sticky, so resetting the deadline does not permit
		// making more progress. We might want to change this in the future, but verify
		// the current behavior for now. If we do change this, we'll want to make sure
		// to do so only for writing the response body, not headers.
		if err := ctl.SetWriteDeadline(time.Now().Add(1 * time.Hour)); err != nil {
			t.Errorf("ctl.SetWriteDeadline() = %v, want nil", err)
		}
		w.Write([]byte("three"))
		if err := ctl.Flush(); err == nil {
			t.Errorf("after resetting deadline: ctl.Flush() = nil, want non-nil")
		}
	}))

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatalf("unexpected connection error: %v", err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	if string(b) != "one" {
		t.Errorf("unexpected body: %q", string(b))
	}
}

func TestResponseControllerSetFutureWriteDeadline(t *testing.T) {
	run(t, testResponseControllerSetFutureWriteDeadline)
}
func testResponseControllerSetFutureWriteDeadline(t *testing.T, mode testMode) {
	errc := make(chan error, 1)
	startwritec := make(chan struct{})
	cst := newClientServerTest(t, mode, HandlerFunc(func { w, r ->
		ctl := NewResponseController(w)
		w.WriteHeader(200)
		if err := ctl.Flush(); err != nil {
			t.Errorf("ctl.Flush() = %v, want nil", err)
		}
		<-startwritec // don't set the deadline until the client reads response headers
		if err := ctl.SetWriteDeadline(time.Now().Add(1 * time.Millisecond)); err != nil {
			t.Errorf("ctl.SetWriteDeadline() = %v, want nil", err)
		}
		_, err := io.Copy(w, neverEnding('a'))
		errc <- err
	}))

	res, err := cst.c.Get(cst.ts.URL)
	close(startwritec)
	if err != nil {
		t.Fatalf("unexpected connection error: %v", err)
	}
	defer res.Body.Close()
	_, err = io.Copy(io.Discard, res.Body)
	if err == nil {
		t.Errorf("client reading from truncated request body: got nil error, want non-nil")
	}
	err = <-errc // io.Copy error
	if !errors.Is(err, os.ErrDeadlineExceeded) {
		t.Errorf("server timed out writing request body: got err %v; want os.ErrDeadlineExceeded", err)
	}
}

func TestResponseControllerSetPastReadDeadline(t *testing.T) {
	run(t, testResponseControllerSetPastReadDeadline)
}
func testResponseControllerSetPastReadDeadline(t *testing.T, mode testMode) {
	readc := make(chan struct{})
	donec := make(chan struct{})
	cst := newClientServerTest(t, mode, HandlerFunc(func { w, r ->
		defer close(donec)
		ctl := NewResponseController(w)
		b := make([]byte, 3)
		n, err := io.ReadFull(r.Body, b)
		b = b[:n]
		if err != nil || string(b) != "one" {
			t.Errorf("before setting read deadline: Read = %v, %q, want nil, %q", err, string(b), "one")
			return
		}
		if err := ctl.SetReadDeadline(time.Now()); err != nil {
			t.Errorf("ctl.SetReadDeadline() = %v, want nil", err)
			return
		}
		b, err = io.ReadAll(r.Body)
		if err == nil || string(b) != "" {
			t.Errorf("after setting read deadline: Read = %q, nil, want error", string(b))
		}
		close(readc)
		// Connection errors are sticky, so resetting the deadline does not permit
		// making more progress. We might want to change this in the future, but verify
		// the current behavior for now.
		if err := ctl.SetReadDeadline(time.Time{}); err != nil {
			t.Errorf("ctl.SetReadDeadline() = %v, want nil", err)
			return
		}
		b, err = io.ReadAll(r.Body)
		if err == nil {
			t.Errorf("after resetting read deadline: Read = %q, nil, want error", string(b))
		}
	}))

	pr, pw := io.Pipe()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer pw.Close()
		pw.Write([]byte("one"))
		select {
		case <-readc:
		case <-donec:
			select {
			case <-readc:
			default:
				t.Errorf("server handler unexpectedly exited without closing readc")
				return
			}
		}
		pw.Write([]byte("two"))
	}()
	defer wg.Wait()
	res, err := cst.c.Post(cst.ts.URL, "text/foo", pr)
	if err == nil {
		defer res.Body.Close()
	}
}

func TestResponseControllerSetFutureReadDeadline(t *testing.T) {
	run(t, testResponseControllerSetFutureReadDeadline)
}
func testResponseControllerSetFutureReadDeadline(t *testing.T, mode testMode) {
	respBody := "response body"
	cst := newClientServerTest(t, mode, HandlerFunc(func { w, req ->
		ctl := NewResponseController(w)
		if err := ctl.SetReadDeadline(time.Now().Add(1 * time.Millisecond)); err != nil {
			t.Errorf("ctl.SetReadDeadline() = %v, want nil", err)
		}
		_, err := io.Copy(io.Discard, req.Body)
		if !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("server timed out reading request body: got err %v; want os.ErrDeadlineExceeded", err)
		}
		w.Write([]byte(respBody))
	}))
	pr, pw := io.Pipe()
	res, err := cst.c.Post(cst.ts.URL, "text/apocryphal", pr)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	got, err := io.ReadAll(res.Body)
	if string(got) != respBody || err != nil {
		t.Errorf("client read response body: %q, %v; want %q, nil", string(got), err, respBody)
	}
	pw.Close()
}

type wrapWriter struct {
	ResponseWriter
}

func (w wrapWriter) Unwrap() ResponseWriter {
	return w.ResponseWriter
}

func TestWrappedResponseController(t *testing.T) { run(t, testWrappedResponseController) }
func testWrappedResponseController(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func { w, r ->
		w = wrapWriter{w}
		ctl := NewResponseController(w)
		if err := ctl.Flush(); err != nil {
			t.Errorf("ctl.Flush() = %v, want nil", err)
		}
		if err := ctl.SetReadDeadline(time.Time{}); err != nil {
			t.Errorf("ctl.SetReadDeadline() = %v, want nil", err)
		}
		if err := ctl.SetWriteDeadline(time.Time{}); err != nil {
			t.Errorf("ctl.SetWriteDeadline() = %v, want nil", err)
		}
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatalf("unexpected connection error: %v", err)
	}
	io.Copy(io.Discard, res.Body)
	defer res.Body.Close()
}

func TestResponseControllerEnableFullDuplex(t *testing.T) {
	run(t, testResponseControllerEnableFullDuplex)
}
func testResponseControllerEnableFullDuplex(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func { w, req ->
		ctl := NewResponseController(w)
		if err := ctl.EnableFullDuplex(); err != nil {
			// TODO: Drop test for HTTP/2 when x/net is updated to support
			// EnableFullDuplex. Since HTTP/2 supports full duplex by default,
			// the rest of the test is fine; it's just the EnableFullDuplex call
			// that fails.
			if mode != http2Mode {
				t.Errorf("ctl.EnableFullDuplex() = %v, want nil", err)
			}
		}
		w.WriteHeader(200)
		ctl.Flush()
		for {
			var buf [1]byte
			n, err := req.Body.Read(buf[:])
			if n != 1 || err != nil {
				break
			}
			w.Write(buf[:])
			ctl.Flush()
		}
	}))
	pr, pw := io.Pipe()
	res, err := cst.c.Post(cst.ts.URL, "text/apocryphal", pr)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	for i := byte(0); i < 10; i++ {
		if _, err := pw.Write([]byte{i}); err != nil {
			t.Fatalf("Write: %v", err)
		}
		var buf [1]byte
		if n, err := res.Body.Read(buf[:]); n != 1 || err != nil {
			t.Fatalf("Read: %v, %v", n, err)
		}
		if buf[0] != i {
			t.Fatalf("read byte %v, want %v", buf[0], i)
		}
	}
	pw.Close()
}

func TestIssue58237(t *testing.T) {
	cst := newClientServerTest(t, http2Mode, HandlerFunc(func { w, req ->
		ctl := NewResponseController(w)
		if err := ctl.SetReadDeadline(time.Now().Add(1 * time.Millisecond)); err != nil {
			t.Errorf("ctl.SetReadDeadline() = %v, want nil", err)
		}
		time.Sleep(10 * time.Millisecond)
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}
