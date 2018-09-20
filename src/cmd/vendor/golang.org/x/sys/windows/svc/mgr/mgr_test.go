// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package mgr_test

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/windows/svc/mgr"
)

func TestOpenLanManServer(t *testing.T) {
	m, err := mgr.Connect()
	if err != nil {
		if errno, ok := err.(syscall.Errno); ok && errno == syscall.ERROR_ACCESS_DENIED {
			t.Skip("Skipping test: we don't have rights to manage services.")
		}
		t.Fatalf("SCM connection failed: %s", err)
	}
	defer m.Disconnect()

	s, err := m.OpenService("LanmanServer")
	if err != nil {
		t.Fatalf("OpenService(lanmanserver) failed: %s", err)
	}
	defer s.Close()

	_, err = s.Config()
	if err != nil {
		t.Fatalf("Config failed: %s", err)
	}
}

func install(t *testing.T, m *mgr.Mgr, name, exepath string, c mgr.Config) {
	// Sometimes it takes a while for the service to get
	// removed after previous test run.
	for i := 0; ; i++ {
		s, err := m.OpenService(name)
		if err != nil {
			break
		}
		s.Close()

		if i > 10 {
			t.Fatalf("service %s already exists", name)
		}
		time.Sleep(300 * time.Millisecond)
	}

	s, err := m.CreateService(name, exepath, c)
	if err != nil {
		t.Fatalf("CreateService(%s) failed: %v", name, err)
	}
	defer s.Close()
}

func depString(d []string) string {
	if len(d) == 0 {
		return ""
	}
	for i := range d {
		d[i] = strings.ToLower(d[i])
	}
	ss := sort.StringSlice(d)
	ss.Sort()
	return strings.Join([]string(ss), " ")
}

func testConfig(t *testing.T, s *mgr.Service, should mgr.Config) mgr.Config {
	is, err := s.Config()
	if err != nil {
		t.Fatalf("Config failed: %s", err)
	}
	if should.DisplayName != is.DisplayName {
		t.Fatalf("config mismatch: DisplayName is %q, but should have %q", is.DisplayName, should.DisplayName)
	}
	if should.StartType != is.StartType {
		t.Fatalf("config mismatch: StartType is %v, but should have %v", is.StartType, should.StartType)
	}
	if should.Description != is.Description {
		t.Fatalf("config mismatch: Description is %q, but should have %q", is.Description, should.Description)
	}
	if depString(should.Dependencies) != depString(is.Dependencies) {
		t.Fatalf("config mismatch: Dependencies is %v, but should have %v", is.Dependencies, should.Dependencies)
	}
	return is
}

func remove(t *testing.T, s *mgr.Service) {
	err := s.Delete()
	if err != nil {
		t.Fatalf("Delete failed: %s", err)
	}
}

func TestMyService(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode - it modifies system services")
	}

	const name = "myservice"

	m, err := mgr.Connect()
	if err != nil {
		if errno, ok := err.(syscall.Errno); ok && errno == syscall.ERROR_ACCESS_DENIED {
			t.Skip("Skipping test: we don't have rights to manage services.")
		}
		t.Fatalf("SCM connection failed: %s", err)
	}
	defer m.Disconnect()

	c := mgr.Config{
		StartType:    mgr.StartDisabled,
		DisplayName:  "my service",
		Description:  "my service is just a test",
		Dependencies: []string{"LanmanServer", "W32Time"},
	}

	exename := os.Args[0]
	exepath, err := filepath.Abs(exename)
	if err != nil {
		t.Fatalf("filepath.Abs(%s) failed: %s", exename, err)
	}

	install(t, m, name, exepath, c)

	s, err := m.OpenService(name)
	if err != nil {
		t.Fatalf("service %s is not installed", name)
	}
	defer s.Close()

	c.BinaryPathName = exepath
	c = testConfig(t, s, c)

	c.StartType = mgr.StartManual
	err = s.UpdateConfig(c)
	if err != nil {
		t.Fatalf("UpdateConfig failed: %v", err)
	}

	testConfig(t, s, c)

	svcnames, err := m.ListServices()
	if err != nil {
		t.Fatalf("ListServices failed: %v", err)
	}
	var myserviceIsInstalled bool
	for _, sn := range svcnames {
		if sn == name {
			myserviceIsInstalled = true
			break
		}
	}
	if !myserviceIsInstalled {
		t.Errorf("ListServices failed to find %q service", name)
	}

	remove(t, s)
}
