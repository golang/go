// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"testing"
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
