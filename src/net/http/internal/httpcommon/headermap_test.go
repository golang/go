// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpcommon

import (
	"net/textproto"
	"testing"
)

func TestCommonHeaders(t *testing.T) {
	for _, lower := range commonHeaders {
		canon := textproto.CanonicalMIMEHeaderKey(lower)
		if got := lowerCommonHeader(canon); got != lower {
			t.Errorf("lowerCommonHeader(%q) = %q, want %q", canon, got, lower)
		}
		if got := canonCommonHeader(lower); got != canon {
			t.Errorf("canonCommonHeader(%q) = %q, want %q", lower, got, canon)
		}
		if got := lowerCommonHeader(lower); got != "" {
			t.Errorf("lowerCommonHeader(%q) = %q, want \"\"", lower, got)
		}
		if got := canonCommonHeader(canon); got != "" {
			t.Errorf("canonCommonHeader(%q) = %q, want \"\"", canon, got)
		}
	}
}

func BenchmarkCanonicalHeader(b *testing.B) {
	for _, name := range []string{
		"content-type",
		"user-agent",
		"x-request-id",
		"x-envoy-upstream-service-time",
	} {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				CanonicalHeader(name)
			}
		})
	}
}

func BenchmarkCachedCanonicalHeader(b *testing.B) {
	for _, name := range []string{
		"content-type",
		"user-agent",
		"x-request-id",
		"x-envoy-upstream-service-time",
	} {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				CachedCanonicalHeader(name)
			}
		})
	}
}
