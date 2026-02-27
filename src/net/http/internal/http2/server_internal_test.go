// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
)

func TestCheckValidHTTP2Request(t *testing.T) {
	tests := []struct {
		h    http.Header
		want error
	}{
		{
			h:    http.Header{"Te": {"trailers"}},
			want: nil,
		},
		{
			h:    http.Header{"Te": {"trailers", "bogus"}},
			want: errors.New(`request header "TE" may only be "trailers" in HTTP/2`),
		},
		{
			h:    http.Header{"Foo": {""}},
			want: nil,
		},
		{
			h:    http.Header{"Connection": {""}},
			want: errors.New(`request header "Connection" is not valid in HTTP/2`),
		},
		{
			h:    http.Header{"Proxy-Connection": {""}},
			want: errors.New(`request header "Proxy-Connection" is not valid in HTTP/2`),
		},
		{
			h:    http.Header{"Keep-Alive": {""}},
			want: errors.New(`request header "Keep-Alive" is not valid in HTTP/2`),
		},
		{
			h:    http.Header{"Upgrade": {""}},
			want: errors.New(`request header "Upgrade" is not valid in HTTP/2`),
		},
	}
	for i, tt := range tests {
		got := checkValidHTTP2RequestHeaders(tt.h)
		if !equalError(got, tt.want) {
			t.Errorf("%d. checkValidHTTP2Request = %v; want %v", i, got, tt.want)
		}
	}
}

// TestCanonicalHeaderCacheGrowth verifies that the canonical header cache
// size is capped to a reasonable level.
func TestCanonicalHeaderCacheGrowth(t *testing.T) {
	for _, size := range []int{1, (1 << 20) - 10} {
		base := strings.Repeat("X", size)
		sc := &serverConn{
			serveG: newGoroutineLock(),
		}
		count := 0
		added := 0
		for added < 10*maxCachedCanonicalHeadersKeysSize {
			h := fmt.Sprintf("%v-%v", base, count)
			c := sc.canonicalHeader(h)
			if len(h) != len(c) {
				t.Errorf("sc.canonicalHeader(%q) = %q, want same length", h, c)
			}
			count++
			added += len(h)
		}
		total := 0
		for k, v := range sc.canonHeader {
			total += len(k) + len(v) + 100
		}
		if total > maxCachedCanonicalHeadersKeysSize {
			t.Errorf("after adding %v ~%v-byte headers, canonHeader cache is ~%v bytes, want <%v", count, size, total, maxCachedCanonicalHeadersKeysSize)
		}
	}
}
