// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package net

import (
	"io/fs"
	"os"
	"testing"
	"time"
)

type nssHostTest struct {
	host      string
	localhost string
	want      hostLookupOrder
}

func nssStr(t *testing.T, s string) *nssConf {
	f, err := os.CreateTemp(t.TempDir(), "nss")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(s); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return parseNSSConfFile(f.Name())
}

// represents a dnsConfig returned by parsing a nonexistent resolv.conf
var defaultResolvConf = &dnsConfig{
	servers:  defaultNS,
	ndots:    1,
	timeout:  5,
	attempts: 2,
	err:      fs.ErrNotExist,
}

func TestConfHostLookupOrder(t *testing.T) {
	// These tests are written for a system with cgo available,
	// without using the netgo tag.
	if netGoBuildTag {
		t.Skip("skipping test because net package built with netgo tag")
	}
	if !cgoAvailable {
		t.Skip("skipping test because cgo resolver not available")
	}

	tests := []struct {
		name      string
		c         *conf
		nss       *nssConf
		resolver  *Resolver
		resolv    *dnsConfig
		hostTests []nssHostTest
	}{
		{
			name: "force",
			c: &conf{
				preferCgo: true,
				netCgo:    true,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "foo: bar"),
			hostTests: []nssHostTest{
				{"foo.local", "myhostname", hostLookupCgo},
				{"google.com", "myhostname", hostLookupCgo},
			},
		},
		{
			name: "netgo_dns_before_files",
			c: &conf{
				netGo: true,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: dns files"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupDNSFiles},
			},
		},
		{
			name: "netgo_fallback_on_cgo",
			c: &conf{
				netGo: true,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: dns files something_custom"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupDNSFiles},
			},
		},
		{
			name: "ubuntu_trusty_avahi",
			c: &conf{
				mdnsTest: mdnsAssumeDoesNotExist,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4"),
			hostTests: []nssHostTest{
				{"foo.local", "myhostname", hostLookupCgo},
				{"foo.local.", "myhostname", hostLookupCgo},
				{"foo.LOCAL", "myhostname", hostLookupCgo},
				{"foo.LOCAL.", "myhostname", hostLookupCgo},
				{"google.com", "myhostname", hostLookupFilesDNS},
			},
		},
		{
			name: "freebsdlinux_no_resolv_conf",
			c: &conf{
				goos: "freebsd",
			},
			resolv:    defaultResolvConf,
			nss:       nssStr(t, "foo: bar"),
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupFilesDNS}},
		},
		// On OpenBSD, no resolv.conf means no DNS.
		{
			name: "openbsd_no_resolv_conf",
			c: &conf{
				goos: "openbsd",
			},
			resolv:    defaultResolvConf,
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupFiles}},
		},
		{
			name: "solaris_no_nsswitch",
			c: &conf{
				goos: "solaris",
			},
			resolv:    defaultResolvConf,
			nss:       &nssConf{err: fs.ErrNotExist},
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupCgo}},
		},
		{
			name: "openbsd_lookup_bind_file",
			c: &conf{
				goos: "openbsd",
			},
			resolv: &dnsConfig{lookup: []string{"bind", "file"}},
			hostTests: []nssHostTest{
				{"google.com", "myhostname", hostLookupDNSFiles},
				{"foo.local", "myhostname", hostLookupDNSFiles},
			},
		},
		{
			name: "openbsd_lookup_file_bind",
			c: &conf{
				goos: "openbsd",
			},
			resolv:    &dnsConfig{lookup: []string{"file", "bind"}},
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupFilesDNS}},
		},
		{
			name: "openbsd_lookup_bind",
			c: &conf{
				goos: "openbsd",
			},
			resolv:    &dnsConfig{lookup: []string{"bind"}},
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupDNS}},
		},
		{
			name: "openbsd_lookup_file",
			c: &conf{
				goos: "openbsd",
			},
			resolv:    &dnsConfig{lookup: []string{"file"}},
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupFiles}},
		},
		{
			name: "openbsd_lookup_yp",
			c: &conf{
				goos: "openbsd",
			},
			resolv:    &dnsConfig{lookup: []string{"file", "bind", "yp"}},
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupCgo}},
		},
		{
			name: "openbsd_lookup_two",
			c: &conf{
				goos: "openbsd",
			},
			resolv:    &dnsConfig{lookup: []string{"file", "foo"}},
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupCgo}},
		},
		{
			name: "openbsd_lookup_empty",
			c: &conf{
				goos: "openbsd",
			},
			resolv:    &dnsConfig{lookup: nil},
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupDNSFiles}},
		},
		{
			name: "linux_no_nsswitch.conf",
			c: &conf{
				goos: "linux",
			},
			resolv:    defaultResolvConf,
			nss:       &nssConf{err: fs.ErrNotExist},
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupFilesDNS}},
		},
		{
			name: "linux_empty_nsswitch.conf",
			c: &conf{
				goos: "linux",
			},
			resolv:    defaultResolvConf,
			nss:       nssStr(t, ""),
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupFilesDNS}},
		},
		{
			name: "files_mdns_dns",
			c: &conf{
				mdnsTest: mdnsAssumeDoesNotExist,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: files mdns dns"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupFilesDNS},
				{"x.local", "myhostname", hostLookupCgo},
			},
		},
		{
			name:   "dns_special_hostnames",
			c:      &conf{},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: dns"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupDNS},
				{"x\\.com", "myhostname", hostLookupCgo},     // punt on weird glibc escape
				{"foo.com%en0", "myhostname", hostLookupCgo}, // and IPv6 zones
			},
		},
		{
			name: "mdns_allow",
			c: &conf{
				mdnsTest: mdnsAssumeExists,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: files mdns dns"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupCgo},
				{"x.local", "myhostname", hostLookupCgo},
			},
		},
		{
			name:   "files_dns",
			c:      &conf{},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: files dns"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupFilesDNS},
				{"x", "myhostname", hostLookupFilesDNS},
				{"x.local", "myhostname", hostLookupFilesDNS},
			},
		},
		{
			name:   "dns_files",
			c:      &conf{},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: dns files"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupDNSFiles},
				{"x", "myhostname", hostLookupDNSFiles},
				{"x.local", "myhostname", hostLookupDNSFiles},
			},
		},
		{
			name:   "something_custom",
			c:      &conf{},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: dns files something_custom"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupCgo},
			},
		},
		{
			name:   "myhostname",
			c:      &conf{},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: files dns myhostname"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupFilesDNS},
				{"myhostname", "myhostname", hostLookupCgo},
				{"myHostname", "myhostname", hostLookupCgo},
				{"myhostname.dot", "myhostname.dot", hostLookupCgo},
				{"myHostname.dot", "myhostname.dot", hostLookupCgo},
				{"_gateway", "myhostname", hostLookupCgo},
				{"_Gateway", "myhostname", hostLookupCgo},
				{"_outbound", "myhostname", hostLookupCgo},
				{"_Outbound", "myhostname", hostLookupCgo},
				{"localhost", "myhostname", hostLookupCgo},
				{"Localhost", "myhostname", hostLookupCgo},
				{"anything.localhost", "myhostname", hostLookupCgo},
				{"Anything.localhost", "myhostname", hostLookupCgo},
				{"localhost.localdomain", "myhostname", hostLookupCgo},
				{"Localhost.Localdomain", "myhostname", hostLookupCgo},
				{"anything.localhost.localdomain", "myhostname", hostLookupCgo},
				{"Anything.Localhost.Localdomain", "myhostname", hostLookupCgo},
				{"somehostname", "myhostname", hostLookupFilesDNS},
			},
		},
		{
			name: "ubuntu14.04.02",
			c: &conf{
				mdnsTest: mdnsAssumeDoesNotExist,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: files myhostname mdns4_minimal [NOTFOUND=return] dns mdns4"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupFilesDNS},
				{"somehostname", "myhostname", hostLookupFilesDNS},
				{"myhostname", "myhostname", hostLookupCgo},
			},
		},
		// Debian Squeeze is just "dns,files", but lists all
		// the default criteria for dns, but then has a
		// non-standard but redundant notfound=return for the
		// files.
		{
			name:   "debian_squeeze",
			c:      &conf{},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: dns [success=return notfound=continue unavail=continue tryagain=continue] files [notfound=return]"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupDNSFiles},
				{"somehostname", "myhostname", hostLookupDNSFiles},
			},
		},
		{
			name:      "resolv.conf-unknown",
			c:         &conf{},
			resolv:    &dnsConfig{servers: defaultNS, ndots: 1, timeout: 5, attempts: 2, unknownOpt: true},
			nss:       nssStr(t, "foo: bar"),
			hostTests: []nssHostTest{{"google.com", "myhostname", hostLookupCgo}},
		},
		// Issue 24393: make sure "Resolver.PreferGo = true" acts like netgo.
		{
			name:     "resolver-prefergo",
			resolver: &Resolver{PreferGo: true},
			c: &conf{
				preferCgo: true,
				netCgo:    true,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, ""),
			hostTests: []nssHostTest{
				{"localhost", "myhostname", hostLookupFilesDNS},
			},
		},
		{
			name:     "unknown-source",
			resolver: &Resolver{PreferGo: true},
			c:        &conf{},
			resolv:   defaultResolvConf,
			nss:      nssStr(t, "hosts: resolve files"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupDNSFiles},
			},
		},
		{
			name:     "dns-among-unknown-sources",
			resolver: &Resolver{PreferGo: true},
			c:        &conf{},
			resolv:   defaultResolvConf,
			nss:      nssStr(t, "hosts: mymachines files dns"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupFilesDNS},
			},
		},
		{
			name:     "dns-among-unknown-sources-2",
			resolver: &Resolver{PreferGo: true},
			c:        &conf{},
			resolv:   defaultResolvConf,
			nss:      nssStr(t, "hosts: dns mymachines files"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupDNSFiles},
			},
		},
	}

	origGetHostname := getHostname
	defer func() { getHostname = origGetHostname }()
	defer setSystemNSS(getSystemNSS(), 0)
	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	for _, tt := range tests {
		if !conf.forceUpdateConf(tt.resolv, time.Now().Add(time.Hour)) {
			t.Errorf("%s: failed to change resolv config", tt.name)
		}
		for _, ht := range tt.hostTests {
			getHostname = func() (string, error) { return ht.localhost, nil }
			setSystemNSS(tt.nss, time.Hour)

			gotOrder, _ := tt.c.hostLookupOrder(tt.resolver, ht.host)
			if gotOrder != ht.want {
				t.Errorf("%s: hostLookupOrder(%q) = %v; want %v", tt.name, ht.host, gotOrder, ht.want)
			}
		}
	}
}

func TestAddrLookupOrder(t *testing.T) {
	// This test is written for a system with cgo available,
	// without using the netgo tag.
	if netGoBuildTag {
		t.Skip("skipping test because net package built with netgo tag")
	}
	if !cgoAvailable {
		t.Skip("skipping test because cgo resolver not available")
	}

	defer setSystemNSS(getSystemNSS(), 0)
	c, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer c.teardown()

	if !c.forceUpdateConf(defaultResolvConf, time.Now().Add(time.Hour)) {
		t.Fatal("failed to change resolv config")
	}

	setSystemNSS(nssStr(t, "hosts: files myhostname dns"), time.Hour)
	cnf := &conf{}
	order, _ := cnf.addrLookupOrder(nil, "192.0.2.1")
	if order != hostLookupCgo {
		t.Errorf("addrLookupOrder returned: %v, want cgo", order)
	}

	setSystemNSS(nssStr(t, "hosts: files mdns4 dns"), time.Hour)
	order, _ = cnf.addrLookupOrder(nil, "192.0.2.1")
	if order != hostLookupCgo {
		t.Errorf("addrLookupOrder returned: %v, want cgo", order)
	}

}

func setSystemNSS(nss *nssConf, addDur time.Duration) {
	nssConfig.mu.Lock()
	nssConfig.nssConf = nss
	nssConfig.mu.Unlock()
	nssConfig.acquireSema()
	nssConfig.lastChecked = time.Now().Add(addDur)
	nssConfig.releaseSema()
}

func TestSystemConf(t *testing.T) {
	systemConf()
}
