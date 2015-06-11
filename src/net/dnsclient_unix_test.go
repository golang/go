// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"sync"
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
	dir  string
	path string
	*resolverConfig
}

func newResolvConfTest() (*resolvConfTest, error) {
	dir, err := ioutil.TempDir("", "go-resolvconftest")
	if err != nil {
		return nil, err
	}
	conf := &resolvConfTest{
		dir:            dir,
		path:           path.Join(dir, "resolv.conf"),
		resolverConfig: &resolvConf,
	}
	conf.initOnce.Do(conf.init)
	return conf, nil
}

func (conf *resolvConfTest) writeAndUpdate(lines []string) error {
	f, err := os.OpenFile(conf.path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	if _, err := f.WriteString(strings.Join(lines, "\n")); err != nil {
		f.Close()
		return err
	}
	f.Close()
	if err := conf.forceUpdate(conf.path); err != nil {
		return err
	}
	return nil
}

func (conf *resolvConfTest) forceUpdate(name string) error {
	dnsConf := dnsReadConfig(name)
	conf.mu.Lock()
	conf.dnsConfig = dnsConf
	conf.mu.Unlock()
	for i := 0; i < 5; i++ {
		if conf.tryAcquireSema() {
			conf.lastChecked = time.Time{}
			conf.releaseSema()
			return nil
		}
	}
	return fmt.Errorf("tryAcquireSema for %s failed", name)
}

func (conf *resolvConfTest) servers() []string {
	conf.mu.RLock()
	servers := conf.dnsConfig.servers
	conf.mu.RUnlock()
	return servers
}

func (conf *resolvConfTest) teardown() error {
	err := conf.forceUpdate("/etc/resolv.conf")
	os.RemoveAll(conf.dir)
	return err
}

var updateResolvConfTests = []struct {
	name    string   // query name
	lines   []string // resolver configuration lines
	servers []string // expected name servers
}{
	{
		name:    "golang.org",
		lines:   []string{"nameserver 8.8.8.8"},
		servers: []string{"8.8.8.8"},
	},
	{
		name:    "",
		lines:   nil, // an empty resolv.conf should use defaultNS as name servers
		servers: defaultNS,
	},
	{
		name:    "www.example.com",
		lines:   []string{"nameserver 8.8.4.4"},
		servers: []string{"8.8.4.4"},
	},
}

func TestUpdateResolvConf(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	for i, tt := range updateResolvConfTests {
		if err := conf.writeAndUpdate(tt.lines); err != nil {
			t.Error(err)
			continue
		}
		if tt.name != "" {
			var wg sync.WaitGroup
			const N = 10
			wg.Add(N)
			for j := 0; j < N; j++ {
				go func(name string) {
					defer wg.Done()
					ips, err := goLookupIP(name)
					if err != nil {
						t.Error(err)
						return
					}
					if len(ips) == 0 {
						t.Errorf("no records for %s", name)
						return
					}
				}(tt.name)
			}
			wg.Wait()
		}
		servers := conf.servers()
		if !reflect.DeepEqual(servers, tt.servers) {
			t.Errorf("#%d: got %v; want %v", i, servers, tt.servers)
			continue
		}
	}
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

	conf, err := newResolvConfTest()
	if err != nil {
		b.Fatal(err)
	}
	defer conf.teardown()

	lines := []string{
		"nameserver 203.0.113.254", // use TEST-NET-3 block, see RFC 5737
		"nameserver 8.8.8.8",
	}
	if err := conf.writeAndUpdate(lines); err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		goLookupIP("www.example.com")
	}
}
