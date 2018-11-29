// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"net/url"
	"os"
	"testing"
)

// TODO(mattn):
//	test ProxyAuth

var cacheKeysTests = []struct {
	proxy  string
	scheme string
	addr   string
	key    string
}{
	{"", "http", "foo.com", "|http|foo.com"},
	{"", "https", "foo.com", "|https|foo.com"},
	{"http://foo.com", "http", "foo.com", "http://foo.com|http|"},
	{"http://foo.com", "https", "foo.com", "http://foo.com|https|foo.com"},
}

func TestCacheKeys(t *testing.T) {
	for _, tt := range cacheKeysTests {
		var proxy *url.URL
		if tt.proxy != "" {
			u, err := url.Parse(tt.proxy)
			if err != nil {
				t.Fatal(err)
			}
			proxy = u
		}
		cm := connectMethod{proxy, tt.scheme, tt.addr, false}
		if got := cm.key().String(); got != tt.key {
			t.Fatalf("{%q, %q, %q} cache key = %q; want %q", tt.proxy, tt.scheme, tt.addr, got, tt.key)
		}
	}
}

func ResetProxyEnv() {
	for _, v := range []string{"HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy", "REQUEST_METHOD"} {
		os.Unsetenv(v)
	}
	ResetCachedEnvironment()
}
