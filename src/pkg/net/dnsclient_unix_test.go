// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"io"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"testing"
	"time"
)

func TestTCPLookup(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	c, err := Dial("tcp", "8.8.8.8:53")
	if err != nil {
		t.Fatalf("Dial failed: %v", err)
	}
	defer c.Close()
	cfg := &dnsConfig{timeout: 10, attempts: 3}
	_, err = exchange(cfg, c, "com.", dnsTypeALL)
	if err != nil {
		t.Fatalf("exchange failed: %v", err)
	}
}

type resolvConfTest struct {
	*testing.T
	dir     string
	path    string
	started bool
	quitc   chan chan struct{}
}

func newResolvConfTest(t *testing.T) *resolvConfTest {
	dir, err := ioutil.TempDir("", "resolvConfTest")
	if err != nil {
		t.Fatalf("could not create temp dir: %v", err)
	}

	// Disable the default loadConfig
	onceLoadConfig.Do(func() {})

	r := &resolvConfTest{
		T:     t,
		dir:   dir,
		path:  path.Join(dir, "resolv.conf"),
		quitc: make(chan chan struct{}),
	}

	return r
}

func (r *resolvConfTest) Start() {
	loadConfig(r.path, 100*time.Millisecond, r.quitc)
	r.started = true
}

func (r *resolvConfTest) SetConf(s string) {
	// Make sure the file mtime will be different once we're done here,
	// even on systems with coarse (1s) mtime resolution.
	time.Sleep(time.Second)

	f, err := os.OpenFile(r.path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		r.Fatalf("failed to create temp file %s: %v", r.path, err)
	}
	if _, err := io.WriteString(f, s); err != nil {
		f.Close()
		r.Fatalf("failed to write temp file: %v", err)
	}
	f.Close()

	if r.started {
		cfg.ch <- struct{}{} // fill buffer
		cfg.ch <- struct{}{} // wait for reload to begin
		cfg.ch <- struct{}{} // wait for reload to complete
	}
}

func (r *resolvConfTest) WantServers(want []string) {
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
	if got := cfg.dnsConfig.servers; !reflect.DeepEqual(got, want) {
		r.Fatalf("Unexpected dns server loaded, got %v want %v", got, want)
	}
}

func (r *resolvConfTest) Close() {
	resp := make(chan struct{})
	r.quitc <- resp
	<-resp
	if err := os.RemoveAll(r.dir); err != nil {
		r.Logf("failed to remove temp dir %s: %v", r.dir, err)
	}
}

func TestReloadResolvConfFail(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	r := newResolvConfTest(t)
	defer r.Close()

	// resolv.conf.tmp does not exist yet
	r.Start()
	if _, err := goLookupIP("golang.org"); err == nil {
		t.Fatal("goLookupIP(missing) succeeded")
	}

	r.SetConf("nameserver 8.8.8.8")
	if _, err := goLookupIP("golang.org"); err != nil {
		t.Fatalf("goLookupIP(missing; good) failed: %v", err)
	}

	// Using a bad resolv.conf while we had a good
	// one before should not update the config
	r.SetConf("")
	if _, err := goLookupIP("golang.org"); err != nil {
		t.Fatalf("goLookupIP(missing; good; bad) failed: %v", err)
	}
}

func TestReloadResolvConfChange(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	r := newResolvConfTest(t)
	defer r.Close()

	r.SetConf("nameserver 8.8.8.8")
	r.Start()

	if _, err := goLookupIP("golang.org"); err != nil {
		t.Fatalf("goLookupIP(good) failed: %v", err)
	}
	r.WantServers([]string{"[8.8.8.8]"})

	// Using a bad resolv.conf when we had a good one
	// before should not update the config
	r.SetConf("")
	if _, err := goLookupIP("golang.org"); err != nil {
		t.Fatalf("goLookupIP(good; bad) failed: %v", err)
	}

	// A new good config should get picked up
	r.SetConf("nameserver 8.8.4.4")
	r.WantServers([]string{"[8.8.4.4]"})
}
