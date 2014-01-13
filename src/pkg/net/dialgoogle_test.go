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
var testIPv6 = flag.Bool("ipv6", false, "assume ipv6 tunnel is present")

func TestResolveGoogle(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	for _, network := range []string{"tcp", "tcp4", "tcp6"} {
		addr, err := ResolveTCPAddr(network, "www.google.com:http")
		if err != nil {
			if (network == "tcp" || network == "tcp4") && !supportsIPv4 {
				t.Logf("ipv4 is not supported: %v", err)
			} else if network == "tcp6" && !supportsIPv6 {
				t.Logf("ipv6 is not supported: %v", err)
			} else {
				t.Errorf("ResolveTCPAddr failed: %v", err)
			}
			continue
		}
		if (network == "tcp" || network == "tcp4") && addr.IP.To4() == nil {
			t.Errorf("got %v; expected an IPv4 address on %v", addr, network)
		} else if network == "tcp6" && (addr.IP.To16() == nil || addr.IP.To4() != nil) {
			t.Errorf("got %v; expected an IPv6 address on %v", addr, network)
		}
	}
}

func TestDialGoogle(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	d := &Dialer{DualStack: true}
	for _, network := range []string{"tcp", "tcp4", "tcp6"} {
		if network == "tcp" && !supportsIPv4 && !supportsIPv6 {
			t.Logf("skipping test; both ipv4 and ipv6 are not supported")
			continue
		} else if network == "tcp4" && !supportsIPv4 {
			t.Logf("skipping test; ipv4 is not supported")
			continue
		} else if network == "tcp6" && !supportsIPv6 {
			t.Logf("skipping test; ipv6 is not supported")
			continue
		} else if network == "tcp6" && !*testIPv6 {
			t.Logf("test disabled; use -ipv6 to enable")
			continue
		}
		if c, err := d.Dial(network, "www.google.com:http"); err != nil {
			t.Errorf("Dial failed: %v", err)
		} else {
			c.Close()
		}
	}
}

// fd is already connected to the destination, port 80.
// Run an HTTP request to fetch the appropriate page.
func fetchGoogle(t *testing.T, fd Conn, network, addr string) {
	req := []byte("GET /robots.txt HTTP/1.0\r\nHost: www.google.com\r\n\r\n")
	n, err := fd.Write(req)

	buf := make([]byte, 1000)
	n, err = io.ReadFull(fd, buf)

	if n < 1000 {
		t.Errorf("fetchGoogle: short HTTP read from %s %s - %v", network, addr, err)
		return
	}
}

func doDial(t *testing.T, network, addr string) {
	fd, err := Dial(network, addr)
	if err != nil {
		t.Errorf("Dial(%q, %q, %q) = _, %v", network, "", addr, err)
		return
	}
	fetchGoogle(t, fd, network, addr)
	fd.Close()
}

var googleaddrsipv4 = []string{
	"%d.%d.%d.%d:80",
	"www.google.com:80",
	"%d.%d.%d.%d:http",
	"www.google.com:http",
	"%03d.%03d.%03d.%03d:0080",
	"[::ffff:%d.%d.%d.%d]:80",
	"[::ffff:%02x%02x:%02x%02x]:80",
	"[0:0:0:0:0000:ffff:%d.%d.%d.%d]:80",
	"[0:0:0:0:000000:ffff:%d.%d.%d.%d]:80",
	"[0:0:0:0::ffff:%d.%d.%d.%d]:80",
}

func TestDialGoogleIPv4(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	// Insert an actual IPv4 address for google.com
	// into the table.
	addrs, err := LookupIP("www.google.com")
	if err != nil {
		t.Fatalf("lookup www.google.com: %v", err)
	}
	var ip IP
	for _, addr := range addrs {
		if x := addr.To4(); x != nil {
			ip = x
			break
		}
	}
	if ip == nil {
		t.Fatalf("no IPv4 addresses for www.google.com")
	}

	for i, s := range googleaddrsipv4 {
		if strings.Contains(s, "%") {
			googleaddrsipv4[i] = fmt.Sprintf(s, ip[0], ip[1], ip[2], ip[3])
		}
	}

	for i := 0; i < len(googleaddrsipv4); i++ {
		addr := googleaddrsipv4[i]
		if addr == "" {
			continue
		}
		t.Logf("-- %s --", addr)
		doDial(t, "tcp", addr)
		if addr[0] != '[' {
			doDial(t, "tcp4", addr)
			if supportsIPv6 {
				// make sure syscall.SocketDisableIPv6 flag works.
				syscall.SocketDisableIPv6 = true
				doDial(t, "tcp", addr)
				doDial(t, "tcp4", addr)
				syscall.SocketDisableIPv6 = false
			}
		}
	}
}

var googleaddrsipv6 = []string{
	"[%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x]:80",
	"ipv6.google.com:80",
	"[%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x]:http",
	"ipv6.google.com:http",
}

func TestDialGoogleIPv6(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	// Only run tcp6 if the kernel will take it.
	if !supportsIPv6 {
		t.Skip("skipping test; ipv6 is not supported")
	}
	if !*testIPv6 {
		t.Skip("test disabled; use -ipv6 to enable")
	}

	// Insert an actual IPv6 address for ipv6.google.com
	// into the table.
	addrs, err := LookupIP("ipv6.google.com")
	if err != nil {
		t.Fatalf("lookup ipv6.google.com: %v", err)
	}
	var ip IP
	for _, addr := range addrs {
		if x := addr.To16(); x != nil {
			ip = x
			break
		}
	}
	if ip == nil {
		t.Fatalf("no IPv6 addresses for ipv6.google.com")
	}

	for i, s := range googleaddrsipv6 {
		if strings.Contains(s, "%") {
			googleaddrsipv6[i] = fmt.Sprintf(s, ip[0], ip[1], ip[2], ip[3], ip[4], ip[5], ip[6], ip[7], ip[8], ip[9], ip[10], ip[11], ip[12], ip[13], ip[14], ip[15])
		}
	}

	for i := 0; i < len(googleaddrsipv6); i++ {
		addr := googleaddrsipv6[i]
		if addr == "" {
			continue
		}
		t.Logf("-- %s --", addr)
		doDial(t, "tcp", addr)
		doDial(t, "tcp6", addr)
	}
}
