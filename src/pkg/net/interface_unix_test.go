// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package net

import (
	"os"
	"os/exec"
	"runtime"
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
		if err := cmd.Run(); err != nil {
			return err
		}
	}
	return nil
}

func (ti *testInterface) teardown() error {
	for _, cmd := range ti.teardownCmds {
		if err := cmd.Run(); err != nil {
			return err
		}
	}
	return nil
}

func TestPointToPointInterface(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	switch {
	case runtime.GOOS == "darwin":
		t.Skipf("skipping read test on %q", runtime.GOOS)
	}
	if os.Getuid() != 0 {
		t.Skip("skipping test; must be root")
	}

	local, remote := "169.254.0.1", "169.254.0.254"
	ip := ParseIP(remote)
	for i := 0; i < 3; i++ {
		ti := &testInterface{}
		if err := ti.setPointToPoint(5963+i, local, remote); err != nil {
			t.Skipf("test requries external command: %v", err)
		}
		if err := ti.setup(); err != nil {
			t.Fatalf("testInterface.setup failed: %v", err)
		} else {
			time.Sleep(3 * time.Millisecond)
		}
		ift, err := Interfaces()
		if err != nil {
			ti.teardown()
			t.Fatalf("Interfaces failed: %v", err)
		}
		for _, ifi := range ift {
			if ti.name == ifi.Name {
				ifat, err := ifi.Addrs()
				if err != nil {
					ti.teardown()
					t.Fatalf("Interface.Addrs failed: %v", err)
				}
				for _, ifa := range ifat {
					if ip.Equal(ifa.(*IPNet).IP) {
						ti.teardown()
						t.Fatalf("got %v; want %v", ip, local)
					}
				}
			}
		}
		if err := ti.teardown(); err != nil {
			t.Fatalf("testInterface.teardown failed: %v", err)
		} else {
			time.Sleep(3 * time.Millisecond)
		}
	}
}

func TestInterfaceArrivalAndDeparture(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	if os.Getuid() != 0 {
		t.Skip("skipping test; must be root")
	}

	for i := 0; i < 3; i++ {
		ift1, err := Interfaces()
		if err != nil {
			t.Fatalf("Interfaces failed: %v", err)
		}
		ti := &testInterface{}
		if err := ti.setBroadcast(5682 + i); err != nil {
			t.Skipf("test requires external command: %v", err)
		}
		if err := ti.setup(); err != nil {
			t.Fatalf("testInterface.setup failed: %v", err)
		} else {
			time.Sleep(3 * time.Millisecond)
		}
		ift2, err := Interfaces()
		if err != nil {
			ti.teardown()
			t.Fatalf("Interfaces failed: %v", err)
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
		if err := ti.teardown(); err != nil {
			t.Fatalf("testInterface.teardown failed: %v", err)
		} else {
			time.Sleep(3 * time.Millisecond)
		}
		ift3, err := Interfaces()
		if err != nil {
			t.Fatalf("Interfaces failed: %v", err)
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
