// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag";
	"io";
	"net";
	"os";
	"syscall";
	"testing";
)

// If an IPv6 tunnel is running (see go/stubl), we can try dialing a real IPv6 address.
var ipv6 = flag.Bool("ipv6", false, "assume ipv6 tunnel is present")

// fd is already connected to the destination, port 80.
// Run an HTTP request to fetch the appropriate page.
func fetchGoogle(t *testing.T, fd net.Conn, network, addr string) {
	req := io.StringBytes("GET /intl/en/privacy.html HTTP/1.0\r\nHost: www.google.com\r\n\r\n");
	n, err := fd.Write(req);

	buf := make([]byte, 1000);
	n, err = io.ReadFull(fd, buf);

	if n < 1000 {
		t.Errorf("fetchGoogle: short HTTP read from %s %s - %v", network, addr, err);
		return
	}
}

func doDial(t *testing.T, network, addr string) {
	fd, err := net.Dial(network, "", addr);
	if err != nil {
		t.Errorf("net.Dial(%q, %q, %q) = _, %v", network, "", addr, err);
		return
	}
	fetchGoogle(t, fd, network, addr);
	fd.Close()
}

func doDialTCP(t *testing.T, network, addr string) {
	fd, err := net.DialTCP(network, "", addr);
	if err != nil {
		t.Errorf("net.DialTCP(%q, %q, %q) = _, %v", network, "", addr, err);
	} else {
		fetchGoogle(t, fd, network, addr);
	}
	fd.Close()
}

var googleaddrs = []string {
	"74.125.19.99:80",
	"www.google.com:80",
	"74.125.19.99:http",
	"www.google.com:http",
	"074.125.019.099:0080",
	"[::ffff:74.125.19.99]:80",
	"[::ffff:4a7d:1363]:80",
	"[0:0:0:0:0000:ffff:74.125.19.99]:80",
	"[0:0:0:0:000000:ffff:74.125.19.99]:80",
	"[0:0:0:0:0:ffff::74.125.19.99]:80",
	"[2001:4860:0:2001::68]:80"	// ipv6.google.com; removed if ipv6 flag not set
}

func TestDialGoogle(t *testing.T) {
	// If no ipv6 tunnel, don't try the last address.
	if !*ipv6 {
		googleaddrs[len(googleaddrs)-1] = ""
	}

	for i := 0; i < len(googleaddrs); i++ {
		addr := googleaddrs[i];
		if addr == "" {
			continue
		}
		t.Logf("-- %s --", addr);
		doDial(t, "tcp", addr);
		doDialTCP(t, "tcp", addr);
		if addr[0] != '[' {
			doDial(t, "tcp4", addr);
			doDialTCP(t, "tcp4", addr);

			if !preferIPv4 {
				// make sure preferIPv4 flag works.
				preferIPv4 = true;
				syscall.SocketDisableIPv6 = true;
				doDial(t, "tcp4", addr);
				doDialTCP(t, "tcp4", addr);
				syscall.SocketDisableIPv6 = false;
				preferIPv4 = false;
			}
		}
		doDial(t, "tcp6", addr);
		doDialTCP(t, "tcp6", addr)
	}
}
