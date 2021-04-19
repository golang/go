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
