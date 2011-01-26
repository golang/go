// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"fmt"
	"io"
	"strings"
	"syscall"
	"testing"
)

// If an IPv6 tunnel is running, we can try dialing a real IPv6 address.
var ipv6 = flag.Bool("ipv6", false, "assume ipv6 tunnel is present")

// fd is already connected to the destination, port 80.
// Run an HTTP request to fetch the appropriate page.
func fetchGoogle(t *testing.T, fd Conn, network, addr string) {
	req := []byte("GET /intl/en/privacy/ HTTP/1.0\r\nHost: www.google.com\r\n\r\n")
	n, err := fd.Write(req)

	buf := make([]byte, 1000)
	n, err = io.ReadFull(fd, buf)

	if n < 1000 {
		t.Errorf("fetchGoogle: short HTTP read from %s %s - %v", network, addr, err)
		return
	}
}

func doDial(t *testing.T, network, addr string) {
	fd, err := Dial(network, "", addr)
	if err != nil {
		t.Errorf("Dial(%q, %q, %q) = _, %v", network, "", addr, err)
		return
	}
	fetchGoogle(t, fd, network, addr)
	fd.Close()
}

var googleaddrs = []string{
	"%d.%d.%d.%d:80",
	"www.google.com:80",
	"%d.%d.%d.%d:http",
	"www.google.com:http",
	"%03d.%03d.%03d.%03d:0080",
	"[::ffff:%d.%d.%d.%d]:80",
	"[::ffff:%02x%02x:%02x%02x]:80",
	"[0:0:0:0:0000:ffff:%d.%d.%d.%d]:80",
	"[0:0:0:0:000000:ffff:%d.%d.%d.%d]:80",
	"[0:0:0:0:0:ffff::%d.%d.%d.%d]:80",
	"[2001:4860:0:2001::68]:80", // ipv6.google.com; removed if ipv6 flag not set
}

func TestDialGoogle(t *testing.T) {
	// If no ipv6 tunnel, don't try the last address.
	if !*ipv6 {
		googleaddrs[len(googleaddrs)-1] = ""
	}

	// Insert an actual IP address for google.com
	// into the table.

	_, addrs, err := LookupHost("www.google.com")
	if err != nil {
		t.Fatalf("lookup www.google.com: %v", err)
	}
	if len(addrs) == 0 {
		t.Fatalf("no addresses for www.google.com")
	}
	ip := ParseIP(addrs[0]).To4()

	for i, s := range googleaddrs {
		if strings.Contains(s, "%") {
			googleaddrs[i] = fmt.Sprintf(s, ip[0], ip[1], ip[2], ip[3])
		}
	}

	for i := 0; i < len(googleaddrs); i++ {
		addr := googleaddrs[i]
		if addr == "" {
			continue
		}
		t.Logf("-- %s --", addr)
		doDial(t, "tcp", addr)
		if addr[0] != '[' {
			doDial(t, "tcp4", addr)

			if !preferIPv4 {
				// make sure preferIPv4 flag works.
				preferIPv4 = true
				syscall.SocketDisableIPv6 = true
				doDial(t, "tcp4", addr)
				syscall.SocketDisableIPv6 = false
				preferIPv4 = false
			}
		}

		// Only run tcp6 if the kernel will take it.
		if kernelSupportsIPv6() {
			doDial(t, "tcp6", addr)
		}
	}
}
