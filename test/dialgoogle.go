// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package main

import (
	"net";
	"flag";
	"io";
	"os";
	"syscall"
)

// If an IPv6 tunnel is running (see go/stubl), we can try dialing a real IPv6 address.
var ipv6 = false
var ipv6_flag = flag.Bool("ipv6", false, &ipv6, "assume ipv6 tunnel is present")

func StringToBuf(s string) *[]byte
{
	l := len(s);
	b := new([]byte, l);
	for i := 0; i < l; i++ {
		b[i] = s[i];
	}
	return b;
}

func Readn(fd io.Read, buf *[]byte) (n int, err *os.Error) {
	n = 0;
	for n < len(buf) {
		nn, e := fd.Read(buf[n:len(buf)]);
		if nn > 0 {
			n += nn
		}
		if e != nil {
			return n, e
		}
	}
	return n, nil
}


// fd is already connected to www.google.com port 80.
// Run an HTTP request to fetch the main page.
func FetchGoogle(fd net.Conn) {
	req := StringToBuf("GET / HTTP/1.0\r\nHost: www.google.com\r\n\r\n");
	n, errno := fd.Write(req);

	buf := new([1000]byte);
	n, errno = Readn(fd, buf);

	fd.Close();
	if n < 1000 {
		panic("short http read");
	}
}

func TestDial(network, addr string) {
	fd, err := net.Dial(network, "", addr);
	if err != nil {
		panic("net.Dial ", network, " ", addr, ": ", err.String())
	}
	FetchGoogle(fd)
}

func TestDialTCP(network, addr string) {
	fd, err := net.DialTCP(network, "", addr);
	if err != nil {
		panic("net.DialTCP ", network, " ", addr, ": ", err.String())
	}
	FetchGoogle(fd)
}

var addrs = []string {
	"74.125.19.99:80",
	"074.125.019.099:0080",
	"[::ffff:74.125.19.99]:80",
	"[::ffff:4a7d:1363]:80",
	"[0:0:0:0:0000:ffff:74.125.19.99]:80",
	"[0:0:0:0:000000:ffff:74.125.19.99]:80",
	"[0:0:0:0:0:ffff::74.125.19.99]:80",
	"[2001:4860:0:2001::68]:80"	// ipv6.google.com; removed if ipv6 flag not set
}

func main()
{
	flag.Parse();
	// If no ipv6 tunnel, don't try the last address.
	if !ipv6 {
		addrs[len(addrs)-1] = ""
	}

	for i := 0; i < len(addrs); i++ {
		addr := addrs[i];
		if addr == "" {
			continue
		}
	//	print(addr, "\n");
		TestDial("tcp", addr);
		TestDialTCP("tcp", addr);
		if addr[0] != '[' {
			TestDial("tcp4", addr);
			TestDialTCP("tcp4", addr)
		}
		TestDial("tcp6", addr);
		TestDialTCP("tcp6", addr)
	}
}
