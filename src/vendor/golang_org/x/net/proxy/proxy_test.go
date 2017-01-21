// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxy

import (
	"io"
	"net"
	"net/url"
	"strconv"
	"sync"
	"testing"
)

func TestFromURL(t *testing.T) {
	endSystem, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen failed: %v", err)
	}
	defer endSystem.Close()
	gateway, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen failed: %v", err)
	}
	defer gateway.Close()

	var wg sync.WaitGroup
	wg.Add(1)
	go socks5Gateway(t, gateway, endSystem, socks5Domain, &wg)

	url, err := url.Parse("socks5://user:password@" + gateway.Addr().String())
	if err != nil {
		t.Fatalf("url.Parse failed: %v", err)
	}
	proxy, err := FromURL(url, Direct)
	if err != nil {
		t.Fatalf("FromURL failed: %v", err)
	}
	_, port, err := net.SplitHostPort(endSystem.Addr().String())
	if err != nil {
		t.Fatalf("net.SplitHostPort failed: %v", err)
	}
	if c, err := proxy.Dial("tcp", "localhost:"+port); err != nil {
		t.Fatalf("FromURL.Dial failed: %v", err)
	} else {
		c.Close()
	}

	wg.Wait()
}

func TestSOCKS5(t *testing.T) {
	endSystem, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen failed: %v", err)
	}
	defer endSystem.Close()
	gateway, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen failed: %v", err)
	}
	defer gateway.Close()

	var wg sync.WaitGroup
	wg.Add(1)
	go socks5Gateway(t, gateway, endSystem, socks5IP4, &wg)

	proxy, err := SOCKS5("tcp", gateway.Addr().String(), nil, Direct)
	if err != nil {
		t.Fatalf("SOCKS5 failed: %v", err)
	}
	if c, err := proxy.Dial("tcp", endSystem.Addr().String()); err != nil {
		t.Fatalf("SOCKS5.Dial failed: %v", err)
	} else {
		c.Close()
	}

	wg.Wait()
}

func socks5Gateway(t *testing.T, gateway, endSystem net.Listener, typ byte, wg *sync.WaitGroup) {
	defer wg.Done()

	c, err := gateway.Accept()
	if err != nil {
		t.Errorf("net.Listener.Accept failed: %v", err)
		return
	}
	defer c.Close()

	b := make([]byte, 32)
	var n int
	if typ == socks5Domain {
		n = 4
	} else {
		n = 3
	}
	if _, err := io.ReadFull(c, b[:n]); err != nil {
		t.Errorf("io.ReadFull failed: %v", err)
		return
	}
	if _, err := c.Write([]byte{socks5Version, socks5AuthNone}); err != nil {
		t.Errorf("net.Conn.Write failed: %v", err)
		return
	}
	if typ == socks5Domain {
		n = 16
	} else {
		n = 10
	}
	if _, err := io.ReadFull(c, b[:n]); err != nil {
		t.Errorf("io.ReadFull failed: %v", err)
		return
	}
	if b[0] != socks5Version || b[1] != socks5Connect || b[2] != 0x00 || b[3] != typ {
		t.Errorf("got an unexpected packet: %#02x %#02x %#02x %#02x", b[0], b[1], b[2], b[3])
		return
	}
	if typ == socks5Domain {
		copy(b[:5], []byte{socks5Version, 0x00, 0x00, socks5Domain, 9})
		b = append(b, []byte("localhost")...)
	} else {
		copy(b[:4], []byte{socks5Version, 0x00, 0x00, socks5IP4})
	}
	host, port, err := net.SplitHostPort(endSystem.Addr().String())
	if err != nil {
		t.Errorf("net.SplitHostPort failed: %v", err)
		return
	}
	b = append(b, []byte(net.ParseIP(host).To4())...)
	p, err := strconv.Atoi(port)
	if err != nil {
		t.Errorf("strconv.Atoi failed: %v", err)
		return
	}
	b = append(b, []byte{byte(p >> 8), byte(p)}...)
	if _, err := c.Write(b); err != nil {
		t.Errorf("net.Conn.Write failed: %v", err)
		return
	}
}
