// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"os"
	"sync"
	"syscall"
	"testing"
	"unsafe"
)

type netlinkAddr struct {
	PID    uint32
	Groups uint32
}

func (a *netlinkAddr) Network() string { return "netlink" }
func (a *netlinkAddr) String() string  { return fmt.Sprintf("%x:%x", a.PID, a.Groups) }

func (a *netlinkAddr) Addr(rsa []byte) Addr {
	if len(rsa) < syscall.SizeofSockaddrNetlink {
		return nil
	}
	var addr netlinkAddr
	b := (*[unsafe.Sizeof(addr)]byte)(unsafe.Pointer(&addr))
	copy(b[0:4], rsa[4:8])
	copy(b[4:8], rsa[8:12])
	return &addr
}

func (a *netlinkAddr) Raw(addr Addr) []byte {
	if addr, ok := addr.(*netlinkAddr); ok {
		rsa := &syscall.RawSockaddrNetlink{Family: syscall.AF_NETLINK, Pid: addr.PID, Groups: addr.Groups}
		return (*[unsafe.Sizeof(*rsa)]byte)(unsafe.Pointer(rsa))[:]
	}
	return nil
}

func TestSocketPacketConn(t *testing.T) {
	s, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_RAW, syscall.NETLINK_ROUTE)
	if err != nil {
		t.Fatal(err)
	}
	lsa := syscall.SockaddrNetlink{Family: syscall.AF_NETLINK}
	if err := syscall.Bind(s, &lsa); err != nil {
		syscall.Close(s)
		t.Fatal(err)
	}
	f := os.NewFile(uintptr(s), "netlink")
	c, err := SocketPacketConn(f, &netlinkAddr{})
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	const N = 3
	var wg sync.WaitGroup
	wg.Add(2 * N)
	dst := &netlinkAddr{PID: 0}
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			l := syscall.NLMSG_HDRLEN + syscall.SizeofRtGenmsg
			b := make([]byte, l)
			*(*uint32)(unsafe.Pointer(&b[0:4][0])) = uint32(l)
			*(*uint16)(unsafe.Pointer(&b[4:6][0])) = uint16(syscall.RTM_GETLINK)
			*(*uint16)(unsafe.Pointer(&b[6:8][0])) = uint16(syscall.NLM_F_DUMP | syscall.NLM_F_REQUEST)
			*(*uint32)(unsafe.Pointer(&b[8:12][0])) = uint32(1)
			*(*uint32)(unsafe.Pointer(&b[12:16][0])) = uint32(0)
			b[16] = byte(syscall.AF_UNSPEC)
			if _, err := c.WriteTo(b, dst); err != nil {
				t.Error(err)
				return
			}
		}()
	}
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			b := make([]byte, os.Getpagesize())
			n, _, err := c.ReadFrom(b)
			if err != nil {
				t.Error(err)
				return
			}
			if _, err := syscall.ParseNetlinkMessage(b[:n]); err != nil {
				t.Error(err)
				return
			}
		}()
	}
	wg.Wait()
}
