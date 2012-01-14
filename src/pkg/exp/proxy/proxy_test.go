// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxy

import (
	"net"
	"net/url"
	"testing"
)

type testDialer struct {
	network, addr string
}

func (t *testDialer) Dial(network, addr string) (net.Conn, error) {
	t.network = network
	t.addr = addr
	return nil, t
}

func (t *testDialer) Error() string {
	return "testDialer " + t.network + " " + t.addr
}

func TestFromURL(t *testing.T) {
	u, err := url.Parse("socks5://user:password@1.2.3.4:5678")
	if err != nil {
		t.Fatalf("failed to parse URL: %s", err)
	}

	tp := &testDialer{}
	proxy, err := FromURL(u, tp)
	if err != nil {
		t.Fatalf("FromURL failed: %s", err)
	}

	conn, err := proxy.Dial("tcp", "example.com:80")
	if conn != nil {
		t.Error("Dial unexpected didn't return an error")
	}
	if tp, ok := err.(*testDialer); ok {
		if tp.network != "tcp" || tp.addr != "1.2.3.4:5678" {
			t.Errorf("Dialer connected to wrong host. Wanted 1.2.3.4:5678, got: %v", tp)
		}
	} else {
		t.Errorf("Unexpected error from Dial: %s", err)
	}
}
