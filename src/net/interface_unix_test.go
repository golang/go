// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package net

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"
)

type testInterface struct {
	name         string
	local        string
	remote       string
	setupCmds    []*exec.Cmd
	teardownCmds []*exec.Cmd
}

func (ti *testInterface) setup() error {
	for _, cmd := range ti.setupCmds {
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("args=%v out=%q err=%v", cmd.Args, string(out), err)
		}
	}
	return nil
}

func (ti *testInterface) teardown() error {
	for _, cmd := range ti.teardownCmds {
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("args=%v out=%q err=%v ", cmd.Args, string(out), err)
		}
	}
	return nil
}

func TestPointToPointInterface(t *testing.T) {
	if testing.Short() {
		t.Skip("avoid external network")
	}
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if os.Getuid() != 0 {
		t.Skip("must be root")
	}

	// We suppose that using IPv4 link-local addresses doesn't
	// harm anyone.
	local, remote := "169.254.0.1", "169.254.0.254"
	ip := ParseIP(remote)
	for i := 0; i < 3; i++ {
		ti := &testInterface{local: local, remote: remote}
		if err := ti.setPointToPoint(5963 + i); err != nil {
			t.Skipf("test requires external command: %v", err)
		}
		if err := ti.setup(); err != nil {
			if e := err.Error(); strings.Contains(e, "No such device") && strings.Contains(e, "gre0") {
				t.Skip("skipping test; no gre0 device. likely running in container?")
			}
			t.Fatal(err)
		} else {
			time.Sleep(3 * time.Millisecond)
		}
		ift, err := Interfaces()
		if err != nil {
			ti.teardown()
			t.Fatal(err)
		}
		for _, ifi := range ift {
			if ti.name != ifi.Name {
				continue
			}
			ifat, err := ifi.Addrs()
			if err != nil {
				ti.teardown()
				t.Fatal(err)
			}
			for _, ifa := range ifat {
				if ip.Equal(ifa.(*IPNet).IP) {
					ti.teardown()
					t.Fatalf("got %v", ifa)
				}
			}
		}
		if err := ti.teardown(); err != nil {
			t.Fatal(err)
		} else {
			time.Sleep(3 * time.Millisecond)
		}
	}
}

func TestInterfaceArrivalAndDeparture(t *testing.T) {
	if testing.Short() {
		t.Skip("avoid external network")
	}
	if os.Getuid() != 0 {
		t.Skip("must be root")
	}

	// We suppose that using IPv4 link-local addresses and the
	// dot1Q ID for Token Ring and FDDI doesn't harm anyone.
	local, remote := "169.254.0.1", "169.254.0.254"
	ip := ParseIP(remote)
	for _, vid := range []int{1002, 1003, 1004, 1005} {
		ift1, err := Interfaces()
		if err != nil {
			t.Fatal(err)
		}
		ti := &testInterface{local: local, remote: remote}
		if err := ti.setBroadcast(vid); err != nil {
			t.Skipf("test requires external command: %v", err)
		}
		if err := ti.setup(); err != nil {
			t.Fatal(err)
		} else {
			time.Sleep(3 * time.Millisecond)
		}
		ift2, err := Interfaces()
		if err != nil {
			ti.teardown()
			t.Fatal(err)
		}
		if len(ift2) <= len(ift1) {
			for _, ifi := range ift1 {
				t.Logf("before: %v", ifi)
			}
			for _, ifi := range ift2 {
				t.Logf("after: %v", ifi)
			}
			ti.teardown()
			t.Fatalf("got %v; want gt %v", len(ift2), len(ift1))
		}
		for _, ifi := range ift2 {
			if ti.name != ifi.Name {
				continue
			}
			ifat, err := ifi.Addrs()
			if err != nil {
				ti.teardown()
				t.Fatal(err)
			}
			for _, ifa := range ifat {
				if ip.Equal(ifa.(*IPNet).IP) {
					ti.teardown()
					t.Fatalf("got %v", ifa)
				}
			}
		}
		if err := ti.teardown(); err != nil {
			t.Fatal(err)
		} else {
			time.Sleep(3 * time.Millisecond)
		}
		ift3, err := Interfaces()
		if err != nil {
			t.Fatal(err)
		}
		if len(ift3) >= len(ift2) {
			for _, ifi := range ift2 {
				t.Logf("before: %v", ifi)
			}
			for _, ifi := range ift3 {
				t.Logf("after: %v", ifi)
			}
			t.Fatalf("got %v; want lt %v", len(ift3), len(ift2))
		}
	}
}

func TestInterfaceArrivalAndDepartureZoneCache(t *testing.T) {
	if testing.Short() {
		t.Skip("avoid external network")
	}
	if os.Getuid() != 0 {
		t.Skip("must be root")
	}

	// Ensure zoneCache is filled:
	_, _ = Listen("tcp", "[fe80::1%nonexistent]:0")

	ti := &testInterface{local: "fe80::1"}
	if err := ti.setLinkLocal(0); err != nil {
		t.Skipf("test requires external command: %v", err)
	}
	if err := ti.setup(); err != nil {
		if e := err.Error(); strings.Contains(e, "Permission denied") {
			t.Skipf("permission denied, skipping test: %v", e)
		}
		t.Fatal(err)
	}
	defer ti.teardown()

	time.Sleep(3 * time.Millisecond)

	// If Listen fails (on Linux with “bind: invalid argument”), zoneCache was
	// not updated when encountering a nonexistent interface:
	ln, err := Listen("tcp", "[fe80::1%"+ti.name+"]:0")
	if err != nil {
		t.Fatal(err)
	}
	ln.Close()
	if err := ti.teardown(); err != nil {
		t.Fatal(err)
	}
}
