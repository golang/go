// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package net

import (
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"
	"unsafe"
)

type routeAddr struct{}

func (a *routeAddr) Network() string { return "route" }
func (a *routeAddr) String() string  { return "<nil>" }

func (a *routeAddr) Addr(rsa []byte) Addr { return &routeAddr{} }
func (a *routeAddr) Raw(addr Addr) []byte { return nil }

func TestSocketConn(t *testing.T) {
	var freebsd32o64 bool
	if runtime.GOOS == "freebsd" && runtime.GOARCH == "386" {
		archs, _ := syscall.Sysctl("kern.supported_archs")
		for _, s := range strings.Split(archs, " ") {
			if strings.TrimSpace(s) == "amd64" {
				freebsd32o64 = true
				break
			}
		}
	}

	s, err := syscall.Socket(syscall.AF_ROUTE, syscall.SOCK_RAW, syscall.AF_UNSPEC)
	if err != nil {
		t.Fatal(err)
	}
	f := os.NewFile(uintptr(s), "route")
	c, err := SocketConn(f, &routeAddr{})
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	const N = 3
	var wg sync.WaitGroup
	wg.Add(2 * N)
	for i := 0; i < N; i++ {
		go func(i int) {
			defer wg.Done()
			l := syscall.SizeofRtMsghdr + syscall.SizeofSockaddrInet4
			if freebsd32o64 {
				l += syscall.SizeofRtMetrics // see syscall/route_freebsd_32bit.go
			}
			b := make([]byte, l)
			h := (*syscall.RtMsghdr)(unsafe.Pointer(&b[0]))
			h.Msglen = uint16(len(b))
			h.Version = syscall.RTM_VERSION
			h.Type = syscall.RTM_GET
			h.Addrs = syscall.RTA_DST
			h.Pid = int32(os.Getpid())
			h.Seq = int32(i)
			p := (*syscall.RawSockaddrInet4)(unsafe.Pointer(&b[syscall.SizeofRtMsghdr]))
			p.Len = syscall.SizeofSockaddrInet4
			p.Family = syscall.AF_INET
			p.Addr = [4]byte{127, 0, 0, 1}
			if _, err := c.Write(b); err != nil {
				t.Error(err)
				return
			}
		}(i + 1)
	}
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			b := make([]byte, os.Getpagesize())
			n, err := c.Read(b)
			if err != nil {
				t.Error(err)
				return
			}
			if _, err := syscall.ParseRoutingMessage(b[:n]); err != nil {
				t.Error(err)
				return
			}
		}()
	}
	wg.Wait()
}
