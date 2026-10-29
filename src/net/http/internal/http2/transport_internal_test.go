// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"testing"
	"time"
)

type panicReader struct{}

func (panicReader) Read([]byte) (int, error) { panic("unexpected Read") }
func (panicReader) Close() error             { panic("unexpected Close") }

func TestActualContentLength(t *testing.T) {
	tests := []struct {
		req  *ClientRequest
		want int64
	}{
		// Verify we don't read from Body:
		0: {
			req:  &ClientRequest{Body: panicReader{}},
			want: -1,
		},
		// nil Body means 0, regardless of ContentLength:
		1: {
			req:  &ClientRequest{Body: nil, ContentLength: 5},
			want: 0,
		},
		// ContentLength is used if set.
		2: {
			req:  &ClientRequest{Body: panicReader{}, ContentLength: 5},
			want: 5,
		},
		// http.NoBody means 0, not -1.
		3: {
			req:  &ClientRequest{Body: NoBody},
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
