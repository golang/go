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

var dnsTransportFallbackTests = []struct {
	server  string
	name    string
	qtype   uint16
	timeout int
	rcode   int
}{
	// Querying "com." with qtype=255 usually makes an answer
	// which requires more than 512 bytes.
	{"8.8.8.8:53", "com.", dnsTypeALL, 2, dnsRcodeSuccess},
	{"8.8.4.4:53", "com.", dnsTypeALL, 4, dnsRcodeSuccess},
}

func TestDNSTransportFallback(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

	for _, tt := range dnsTransportFallbackTests {
		timeout := time.Duration(tt.timeout) * time.Second
		msg, err := exchange(tt.server, tt.name, tt.qtype, timeout)
		if err != nil {
			t.Error(err)
			continue
		}
		switch msg.rcode {
		case tt.rcode, dnsRcodeServerFailure:
		default:
			t.Errorf("got %v from %v; want %v", msg.rcode, tt.server, tt.rcode)
			continue
		}
	}
}

// See RFC 6761 for further information about the reserved, pseudo
// domain names.
var specialDomainNameTests = []struct {
	name  string
	qtype uint16
	rcode int
}{
	// Name resolution APIs and libraries should not recognize the
	// followings as special.
	{"1.0.168.192.in-addr.arpa.", dnsTypePTR, dnsRcodeNameError},
	{"test.", dnsTypeALL, dnsRcodeNameError},
	{"example.com.", dnsTypeALL, dnsRcodeSuccess},

	// Name resolution APIs and libraries should recognize the
	// followings as special and should not send any queries.
	// Though, we test those names here for verifying nagative
	// answers at DNS query-response interaction level.
	{"localhost.", dnsTypeALL, dnsRcodeNameError},
	{"invalid.", dnsTypeALL, dnsRcodeNameError},
}

func TestSpecialDomainName(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

	server := "8.8.8.8:53"
	for _, tt := range specialDomainNameTests {
		msg, err := exchange(server, tt.name, tt.qtype, 0)
		if err != nil {
			t.Error(err)
			continue
		}
		switch msg.rcode {
		case tt.rcode, dnsRcodeServerFailure:
		default:
			t.Errorf("got %v from %v; want %v", msg.rcode, server, tt.rcode)
			continue
		}
	}
}

type resolvConfTest struct {
	*testing.T
	dir  string
	path string
}

func newResolvConfTest(t *testing.T) *resolvConfTest {
	dir, err := ioutil.TempDir("", "go-resolvconftest")
	if err != nil {
		t.Fatal(err)
	}

	r := &resolvConfTest{
		T:    t,
		dir:  dir,
		path: path.Join(dir, "resolv.conf"),
	}

	return r
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
	cfg.lastChecked = time.Time{}
	loadConfig(r.path)
}

func (r *resolvConfTest) WantServers(want []string) {
	cfg.mu.RLock()
	defer cfg.mu.RUnlock()
	if got := cfg.dnsConfig.servers; !reflect.DeepEqual(got, want) {
		r.Fatalf("unexpected dns server loaded, got %v want %v", got, want)
	}
}

func (r *resolvConfTest) Close() {
	if err := os.RemoveAll(r.dir); err != nil {
		r.Logf("failed to remove temp dir %s: %v", r.dir, err)
	}
}

func TestReloadResolvConfFail(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

	r := newResolvConfTest(t)
	defer r.Close()

	r.SetConf("nameserver 8.8.8.8")

	if _, err := goLookupIP("golang.org"); err != nil {
		t.Fatal(err)
	}

	// Using an empty resolv.conf should use localhost as servers
	r.SetConf("")

	if len(cfg.dnsConfig.servers) != len(defaultNS) {
		t.Fatalf("goLookupIP(missing; good; bad) failed: servers=%v, want: %v", cfg.dnsConfig.servers, defaultNS)
	}

	for i := range cfg.dnsConfig.servers {
		if cfg.dnsConfig.servers[i] != defaultNS[i] {
			t.Fatalf("goLookupIP(missing; good; bad) failed: servers=%v, want: %v", cfg.dnsConfig.servers, defaultNS)
		}
	}
}

func TestReloadResolvConfChange(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

	r := newResolvConfTest(t)
	defer r.Close()

	r.SetConf("nameserver 8.8.8.8")

	if _, err := goLookupIP("golang.org"); err != nil {
		t.Fatal(err)
	}
	r.WantServers([]string{"8.8.8.8"})

	// Using an empty resolv.conf should use localhost as servers
	r.SetConf("")

	if len(cfg.dnsConfig.servers) != len(defaultNS) {
		t.Fatalf("goLookupIP(missing; good; bad) failed: servers=%v, want: %v", cfg.dnsConfig.servers, defaultNS)
	}

	for i := range cfg.dnsConfig.servers {
		if cfg.dnsConfig.servers[i] != defaultNS[i] {
			t.Fatalf("goLookupIP(missing; good; bad) failed: servers=%v, want: %v", cfg.dnsConfig.servers, defaultNS)
		}
	}

	// A new good config should get picked up
	r.SetConf("nameserver 8.8.4.4")
	r.WantServers([]string{"8.8.4.4"})
}

func BenchmarkGoLookupIP(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	for i := 0; i < b.N; i++ {
		goLookupIP("www.example.com")
	}
}

func BenchmarkGoLookupIPNoSuchHost(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	for i := 0; i < b.N; i++ {
		goLookupIP("some.nonexistent")
	}
}

func BenchmarkGoLookupIPWithBrokenNameServer(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	// This looks ugly but it's safe as long as benchmarks are run
	// sequentially in package testing.
	<-cfg.ch // keep config from being reloaded upon lookup
	orig := cfg.dnsConfig
	cfg.dnsConfig.servers = append([]string{"203.0.113.254"}, cfg.dnsConfig.servers...) // use TEST-NET-3 block, see RFC 5737
	for i := 0; i < b.N; i++ {
		goLookupIP("www.example.com")
	}
	cfg.dnsConfig = orig
	cfg.ch <- struct{}{}
}
