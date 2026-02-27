// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bytes"
	"compress/gzip"
	"crypto/tls"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

type panicReader struct{}

func (panicReader) Read([]byte) (int, error) { panic("unexpected Read") }
func (panicReader) Close() error             { panic("unexpected Close") }

func TestActualContentLength(t *testing.T) {
	tests := []struct {
		req  *http.Request
		want int64
	}{
		// Verify we don't read from Body:
		0: {
			req:  &http.Request{Body: panicReader{}},
			want: -1,
		},
		// nil Body means 0, regardless of ContentLength:
		1: {
			req:  &http.Request{Body: nil, ContentLength: 5},
			want: 0,
		},
		// ContentLength is used if set.
		2: {
			req:  &http.Request{Body: panicReader{}, ContentLength: 5},
			want: 5,
		},
		// http.NoBody means 0, not -1.
		3: {
			req:  &http.Request{Body: http.NoBody},
			want: 0,
		},
	}
	for i, tt := range tests {
		got := actualContentLength(tt.req)
		if got != tt.want {
			t.Errorf("test[%d]: got %d; want %d", i, got, tt.want)
		}
	}
}

// Tests that gzipReader doesn't crash on a second Read call following
// the first Read call's gzip.NewReader returning an error.
func TestGzipReader_DoubleReadCrash(t *testing.T) {
	gz := &gzipReader{
		body: io.NopCloser(strings.NewReader("0123456789")),
	}
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

func TestGzipReader_ReadAfterClose(t *testing.T) {
	body := bytes.Buffer{}
	w := gzip.NewWriter(&body)
	w.Write([]byte("012345679"))
	w.Close()
	gz := &gzipReader{
		body: io.NopCloser(&body),
	}
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

func TestTransportNewTLSConfig(t *testing.T) {
	tests := [...]struct {
		conf *tls.Config
		host string
		want *tls.Config
	}{
		// Normal case.
		0: {
			conf: nil,
			host: "foo.com",
			want: &tls.Config{
				ServerName: "foo.com",
				NextProtos: []string{NextProtoTLS},
			},
		},

		// User-provided name (bar.com) takes precedence:
		1: {
			conf: &tls.Config{
				ServerName: "bar.com",
			},
			host: "foo.com",
			want: &tls.Config{
				ServerName: "bar.com",
				NextProtos: []string{NextProtoTLS},
			},
		},

		// NextProto is prepended:
		2: {
			conf: &tls.Config{
				NextProtos: []string{"foo", "bar"},
			},
			host: "example.com",
			want: &tls.Config{
				ServerName: "example.com",
				NextProtos: []string{NextProtoTLS, "foo", "bar"},
			},
		},

		// NextProto is not duplicated:
		3: {
			conf: &tls.Config{
				NextProtos: []string{"foo", "bar", NextProtoTLS},
			},
			host: "example.com",
			want: &tls.Config{
				ServerName: "example.com",
				NextProtos: []string{"foo", "bar", NextProtoTLS},
			},
		},
	}
	for i, tt := range tests {
		// Ignore the session ticket keys part, which ends up populating
		// unexported fields in the Config:
		if tt.conf != nil {
			tt.conf.SessionTicketsDisabled = true
		}

		tr := &Transport{TLSClientConfig: tt.conf}
		got := tr.newTLSConfig(tt.host)

		got.SessionTicketsDisabled = false

		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%d. got %#v; want %#v", i, got, tt.want)
		}
	}
}

func TestAuthorityAddr(t *testing.T) {
	tests := []struct {
		scheme, authority string
		want              string
	}{
		{"http", "foo.com", "foo.com:80"},
		{"https", "foo.com", "foo.com:443"},
		{"https", "foo.com:", "foo.com:443"},
		{"https", "foo.com:1234", "foo.com:1234"},
		{"https", "1.2.3.4:1234", "1.2.3.4:1234"},
		{"https", "1.2.3.4", "1.2.3.4:443"},
		{"https", "1.2.3.4:", "1.2.3.4:443"},
		{"https", "[::1]:1234", "[::1]:1234"},
		{"https", "[::1]", "[::1]:443"},
		{"https", "[::1]:", "[::1]:443"},
	}
	for _, tt := range tests {
		got := authorityAddr(tt.scheme, tt.authority)
		if got != tt.want {
			t.Errorf("authorityAddr(%q, %q) = %q; want %q", tt.scheme, tt.authority, got, tt.want)
		}
	}
}

// Issue 25009: use Request.GetBody if present, even if it seems like
// we might not need it. Apparently something else can still read from
// the original request body. Data race? In any case, rewinding
// unconditionally on retry is a nicer model anyway and should
// simplify code in the future (after the Go 1.11 freeze)
func TestTransportUsesGetBodyWhenPresent(t *testing.T) {
	calls := 0
	someBody := func() io.ReadCloser {
		return struct{ io.ReadCloser }{io.NopCloser(bytes.NewReader(nil))}
	}
	req := &http.Request{
		Body: someBody(),
		GetBody: func() (io.ReadCloser, error) {
			calls++
			return someBody(), nil
		},
	}

	req2, err := shouldRetryRequest(req, errClientConnUnusable)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Errorf("Calls = %d; want 1", calls)
	}
	if req2 == req {
		t.Error("req2 changed")
	}
	if req2 == nil {
		t.Fatal("req2 is nil")
	}
	if req2.Body == nil {
		t.Fatal("req2.Body is nil")
	}
	if req2.GetBody == nil {
		t.Fatal("req2.GetBody is nil")
	}
	if req2.Body == req.Body {
		t.Error("req2.Body unchanged")
	}
}

// Issue 22891: verify that the "https" altproto we register with net/http
// is a certain type: a struct with one field with our *http2.Transport in it.
func TestNoDialH2RoundTripperType(t *testing.T) {
	t1 := new(http.Transport)
	t2 := new(Transport)
	rt := noDialH2RoundTripper{t2}
	if err := registerHTTPSProtocol(t1, rt); err != nil {
		t.Fatal(err)
	}
	rv := reflect.ValueOf(rt)
	if rv.Type().Kind() != reflect.Struct {
		t.Fatalf("kind = %v; net/http expects struct", rv.Type().Kind())
	}
	if n := rv.Type().NumField(); n != 1 {
		t.Fatalf("fields = %d; net/http expects 1", n)
	}
	v := rv.Field(0)
	if _, ok := v.Interface().(*Transport); !ok {
		t.Fatalf("wrong kind %T; want *Transport", v.Interface())
	}
}

func TestClientConnTooIdle(t *testing.T) {
	tests := []struct {
		cc   func() *ClientConn
		want bool
	}{
		{
			func() *ClientConn {
				return &ClientConn{idleTimeout: 5 * time.Second, lastIdle: time.Now().Add(-10 * time.Second)}
			},
			true,
		},
		{
			func() *ClientConn {
				return &ClientConn{idleTimeout: 5 * time.Second, lastIdle: time.Time{}}
			},
			false,
		},
		{
			func() *ClientConn {
				return &ClientConn{idleTimeout: 60 * time.Second, lastIdle: time.Now().Add(-10 * time.Second)}
			},
			false,
		},
		{
			func() *ClientConn {
				return &ClientConn{idleTimeout: 0, lastIdle: time.Now().Add(-10 * time.Second)}
			},
			false,
		},
	}
	for i, tt := range tests {
		got := tt.cc().tooIdleLocked()
		if got != tt.want {
			t.Errorf("%d. got %v; want %v", i, got, tt.want)
		}
	}
}
