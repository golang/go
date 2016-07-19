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

func TestMonitorAndParseRIB(t *testing.T) {
	if testing.Short() || os.Getuid() != 0 {
		t.Skip("must be root")
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

	s, err := syscall.Socket(syscall.AF_ROUTE, syscall.SOCK_RAW, syscall.AF_UNSPEC)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Close(s)

	go func() {
		b := make([]byte, os.Getpagesize())
		for {
			n, err := syscall.Read(s, b)
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
