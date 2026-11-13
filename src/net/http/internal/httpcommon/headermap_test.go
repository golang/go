// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpcommon

import "testing"

func TestCommonHeaders(t *testing.T) {
	for _, test := range []struct{ in, lower, canon string }{
		{"", "", ""},
		{"Content-Type", "content-type", ""},
		{"content-type", "", "Content-Type"},
		{"Www-Authenticate", "www-authenticate", ""},
		{"www-authenticate", "", "Www-Authenticate"},
		{"X-Request-Id", "", ""},
		{"x-request-id", "", ""},
	} {
		if got := lowerCommonHeader(test.in); got != test.lower {
			t.Errorf("lowerCommonHeader(%q) = %q, want %q", test.in, got, test.lower)
		}
		if got := canonCommonHeader(test.in); got != test.canon {
			t.Errorf("canonCommonHeader(%q) = %q, want %q", test.in, got, test.canon)
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
