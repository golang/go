// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package route

import (
	"os"
	"syscall"
	"testing"
	"time"
)

func TestFetchAndParseRIB(t *testing.T) {
	for _, af := range []int{sysAF_UNSPEC, sysAF_INET, sysAF_INET6} {
		for _, typ := range []RIBType{sysNET_RT_DUMP, sysNET_RT_IFLIST} {
			ms, err := fetchAndParseRIB(af, typ)
			if err != nil {
				t.Error(err)
				continue
			}
			ss, err := msgs(ms).validate()
			if err != nil {
				t.Errorf("%v %d %v", addrFamily(af), typ, err)
				continue
			}
			for _, s := range ss {
				t.Log(s)
			}
		}
	}
}

var (
	rtmonSock int
	rtmonErr  error
)

func init() {
	// We need to keep rtmonSock alive to avoid treading on
	// recycled socket descriptors.
	rtmonSock, rtmonErr = syscall.Socket(sysAF_ROUTE, sysSOCK_RAW, sysAF_UNSPEC)
}

// TestMonitorAndParseRIB leaks a worker goroutine and a socket
// descriptor but that's intentional.
func TestMonitorAndParseRIB(t *testing.T) {
	if testing.Short() || os.Getuid() != 0 {
		t.Skip("must be root")
	}

	if rtmonErr != nil {
		t.Fatal(rtmonErr)
	}

	// We suppose that using an IPv4 link-local address and the
	// dot1Q ID for Token Ring and FDDI doesn't harm anyone.
	pv := &propVirtual{addr: "169.254.0.1", mask: "255.255.255.0"}
	if err := pv.configure(1002); err != nil {
		t.Skip(err)
	}
	if err := pv.setup(); err != nil {
		t.Skip(err)
	}
	pv.teardown()

	go func() {
		b := make([]byte, os.Getpagesize())
		for {
			// There's no easy way to unblock this read
			// call because the routing message exchange
			// over routing socket is a connectionless
			// message-oriented protocol, no control plane
			// for signaling connectivity, and we cannot
			// use the net package of standard library due
			// to the lack of support for routing socket
			// and circular dependency.
			n, err := syscall.Read(rtmonSock, b)
			if err != nil {
				return
			}
			ms, err := ParseRIB(0, b[:n])
			if err != nil {
				t.Error(err)
				return
			}
			ss, err := msgs(ms).validate()
			if err != nil {
				t.Error(err)
				return
			}
			for _, s := range ss {
				t.Log(s)
			}
		}
	}()

	for _, vid := range []int{1002, 1003, 1004, 1005} {
		pv := &propVirtual{addr: "169.254.0.1", mask: "255.255.255.0"}
		if err := pv.configure(vid); err != nil {
			t.Fatal(err)
		}
		if err := pv.setup(); err != nil {
			t.Fatal(err)
		}
		time.Sleep(200 * time.Millisecond)
		if err := pv.teardown(); err != nil {
			t.Fatal(err)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func TestParseRIBWithFuzz(t *testing.T) {
	for _, fuzz := range []string{
		"0\x00\x05\x050000000000000000" +
			"00000000000000000000" +
			"00000000000000000000" +
			"00000000000000000000" +
			"0000000000000\x02000000" +
			"00000000",
		"\x02\x00\x05\f0000000000000000" +
			"0\x0200000000000000",
		"\x02\x00\x05\x100000000000000\x1200" +
			"0\x00\xff\x00",
		"\x02\x00\x05\f0000000000000000" +
			"0\x12000\x00\x02\x0000",
		"\x00\x00\x00\x01\x00",
		"00000",
	} {
		for typ := RIBType(0); typ < 256; typ++ {
			ParseRIB(typ, []byte(fuzz))
		}
	}
}

func TestRouteMessage(t *testing.T) {
	s, err := syscall.Socket(sysAF_ROUTE, sysSOCK_RAW, sysAF_UNSPEC)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Close(s)

	var ms []RouteMessage
	for _, af := range []int{sysAF_INET, sysAF_INET6} {
		rs, err := fetchAndParseRIB(af, sysNET_RT_DUMP)
		if err != nil || len(rs) == 0 {
			continue
		}
		switch af {
		case sysAF_INET:
			ms = append(ms, []RouteMessage{
				{
					Type: sysRTM_GET,
					Addrs: []Addr{
						&Inet4Addr{IP: [4]byte{127, 0, 0, 1}},
						nil,
						nil,
						nil,
						&LinkAddr{},
						&Inet4Addr{},
						nil,
						&Inet4Addr{},
					},
				},
				{
					Type: sysRTM_GET,
					Addrs: []Addr{
						&Inet4Addr{IP: [4]byte{127, 0, 0, 1}},
					},
				},
			}...)
		case sysAF_INET6:
			ms = append(ms, []RouteMessage{
				{
					Type: sysRTM_GET,
					Addrs: []Addr{
						&Inet6Addr{IP: [16]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}},
						nil,
						nil,
						nil,
						&LinkAddr{},
						&Inet6Addr{},
						nil,
						&Inet6Addr{},
					},
				},
				{
					Type: sysRTM_GET,
					Addrs: []Addr{
						&Inet6Addr{IP: [16]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}},
					},
				},
			}...)
		}
	}
	for i, m := range ms {
		m.ID = uintptr(os.Getpid())
		m.Seq = i + 1
		wb, err := m.Marshal()
		if err != nil {
			t.Fatalf("%v: %v", m, err)
		}
		if _, err := syscall.Write(s, wb); err != nil {
			t.Fatalf("%v: %v", m, err)
		}
		rb := make([]byte, os.Getpagesize())
		n, err := syscall.Read(s, rb)
		if err != nil {
			t.Fatalf("%v: %v", m, err)
		}
		rms, err := ParseRIB(0, rb[:n])
		if err != nil {
			t.Fatalf("%v: %v", m, err)
		}
		for _, rm := range rms {
			err := rm.(*RouteMessage).Err
			if err != nil {
				t.Errorf("%v: %v", m, err)
			}
		}
		ss, err := msgs(rms).validate()
		if err != nil {
			t.Fatalf("%v: %v", m, err)
		}
		for _, s := range ss {
			t.Log(s)
		}

	}
}
