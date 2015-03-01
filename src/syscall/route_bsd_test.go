// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package syscall_test

import (
	"fmt"
	"net"
	"os"
	"syscall"
	"testing"
	"time"
)

func TestRouteRIB(t *testing.T) {
	for _, facility := range []int{syscall.NET_RT_DUMP, syscall.NET_RT_IFLIST} {
		for _, param := range []int{syscall.AF_UNSPEC, syscall.AF_INET, syscall.AF_INET6} {
			var err error
			var b []byte
			// The VM allocator wrapper functions can
			// return ENOMEM easily.
			for i := 0; i < 3; i++ {
				b, err = syscall.RouteRIB(facility, param)
				if err != nil {
					time.Sleep(5 * time.Millisecond)
					continue
				}
				break
			}
			if err != nil {
				t.Error(facility, param, err)
				continue
			}
			msgs, err := syscall.ParseRoutingMessage(b)
			if err != nil {
				t.Error(facility, param, err)
				continue
			}
			var ipv4loopback, ipv6loopback bool
			for _, m := range msgs {
				flags, err := parseRoutingMessageHeader(m)
				if err != nil {
					t.Error(err)
					continue
				}
				sas, err := parseRoutingSockaddrs(m)
				if err != nil {
					t.Error(err)
					continue
				}
				if flags&(syscall.RTA_DST|syscall.RTA_IFA) != 0 {
					sa := sas[syscall.RTAX_DST]
					if sa == nil {
						sa = sas[syscall.RTAX_IFA]
					}
					switch sa := sa.(type) {
					case *syscall.SockaddrInet4:
						if net.IP(sa.Addr[:]).IsLoopback() {
							ipv4loopback = true
						}
					case *syscall.SockaddrInet6:
						if net.IP(sa.Addr[:]).IsLoopback() {
							ipv6loopback = true
						}
					}
				}
				t.Log(facility, param, flags, sockaddrs(sas))
			}
			if param == syscall.AF_UNSPEC && len(msgs) > 0 && !ipv4loopback && !ipv6loopback {
				t.Errorf("no loopback facility found: ipv4/ipv6=%v/%v, %v", ipv4loopback, ipv6loopback, len(msgs))
				continue
			}
		}
	}
}

func TestRouteMonitor(t *testing.T) {
	if testing.Short() || os.Getuid() != 0 {
		t.Skip("must be root")
	}

	s, err := syscall.Socket(syscall.AF_ROUTE, syscall.SOCK_RAW, syscall.AF_UNSPEC)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Close(s)

	tmo := time.After(30 * time.Second)
	go func() {
		b := make([]byte, os.Getpagesize())
		for {
			n, err := syscall.Read(s, b)
			if err != nil {
				return
			}
			msgs, err := syscall.ParseRoutingMessage(b[:n])
			if err != nil {
				t.Error(err)
				return
			}
			for _, m := range msgs {
				flags, err := parseRoutingMessageHeader(m)
				if err != nil {
					t.Error(err)
					continue
				}
				sas, err := parseRoutingSockaddrs(m)
				if err != nil {
					t.Error(err)
					continue
				}
				t.Log(flags, sockaddrs(sas))
			}
		}
	}()
	<-tmo
}

type addrFamily byte

func (f addrFamily) String() string {
	switch f {
	case syscall.AF_UNSPEC:
		return "unspec"
	case syscall.AF_LINK:
		return "link"
	case syscall.AF_INET:
		return "inet4"
	case syscall.AF_INET6:
		return "inet6"
	default:
		return fmt.Sprintf("unknown %d", f)
	}
}

type addrFlags uint32

var addrFlagNames = [...]string{
	"dst",
	"gateway",
	"netmask",
	"genmask",
	"ifp",
	"ifa",
	"author",
	"brd",
	"mpls1,tag,src", // sockaddr_mpls=dragonfly,netbsd, sockaddr_in/in6=openbsd
	"mpls2,srcmask", // sockaddr_mpls=dragonfly, sockaddr_in/in6=openbsd
	"mpls3,label",   // sockaddr_mpls=dragonfly, sockaddr_rtlabel=openbsd
}

func (f addrFlags) String() string {
	var s string
	for i, name := range addrFlagNames {
		if f&(1<<uint(i)) != 0 {
			if s != "" {
				s += "|"
			}
			s += name
		}
	}
	if s == "" {
		return "<nil>"
	}
	return s
}

type sockaddrs []syscall.Sockaddr

func (sas sockaddrs) String() string {
	var s string
	for _, sa := range sas {
		if sa == nil {
			continue
		}
		if len(s) > 0 {
			s += " "
		}
		switch sa := sa.(type) {
		case *syscall.SockaddrDatalink:
			s += fmt.Sprintf("[%v/%v/%v t/n/a/s=%v/%v/%v/%v]", sa.Len, addrFamily(sa.Family), sa.Index, sa.Type, sa.Nlen, sa.Alen, sa.Slen)
		case *syscall.SockaddrInet4:
			s += fmt.Sprintf("%v", net.IP(sa.Addr[:]).To4())
		case *syscall.SockaddrInet6:
			s += fmt.Sprintf("%v", net.IP(sa.Addr[:]).To16())
		}
	}
	if s == "" {
		return "<nil>"
	}
	return s
}

func (sas sockaddrs) match(flags addrFlags) error {
	var f addrFlags
	family := syscall.AF_UNSPEC
	for i := range sas {
		if sas[i] != nil {
			f |= 1 << uint(i)
		}
		switch sas[i].(type) {
		case *syscall.SockaddrInet4:
			if family == syscall.AF_UNSPEC {
				family = syscall.AF_INET
			}
			if family != syscall.AF_INET {
				return fmt.Errorf("got %v; want %v", sockaddrs(sas), family)
			}
		case *syscall.SockaddrInet6:
			if family == syscall.AF_UNSPEC {
				family = syscall.AF_INET6
			}
			if family != syscall.AF_INET6 {
				return fmt.Errorf("got %v; want %v", sockaddrs(sas), family)
			}
		}
	}
	if f != flags {
		return fmt.Errorf("got %v; want %v", f, flags)
	}
	return nil
}
