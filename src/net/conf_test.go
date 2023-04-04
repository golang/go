// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package net

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"runtime"
	"slices"
	"testing"
	"time"

	"maps"
)

type nssHostTest struct {
	host      string
	localhost string
	want      hostLookupOrder
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
				resolver: resolverCgo,
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
				resolver: resolverGo,
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
				resolver: resolverGo,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, "hosts: dns files something_custom"),
			hostTests: []nssHostTest{
				{"x.com", "myhostname", hostLookupFilesDNS},
			},
		},
		{
			name:   "ubuntu_trusty_avahi",
			c:      &conf{},
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
			name:   "files_mdns_dns",
			c:      &conf{},
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
				hasMDNSAllow: true,
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
				{"x.local", "myhostname", hostLookupCgo},
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
				{"x.local", "myhostname", hostLookupCgo},
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
				{"", "myhostname", hostLookupFilesDNS}, // Issue 13623
			},
		},
		{
			name:   "ubuntu14.04.02",
			c:      &conf{},
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
				goos:     "darwin",
				resolver: resolverCgo,
			},
			resolv: defaultResolvConf,
			nss:    nssStr(t, ""),
			hostTests: []nssHostTest{
				{"localhost", "myhostname", hostLookupFilesDNS},
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

type lookupOrderTest struct {
	name string

	mdnsAllow bool
	resolver  *Resolver
	resolv    *dnsConfig
	nss       *nssConf

	// hostname to be queried.
	host string

	// hostname of the system, defaults to "myhostname".
	localhost string

	expect     map[resolverType]hostLookupOrder
	expectGOOS map[string]map[resolverType]hostLookupOrder
}

func testConfHostLookupOrderNsswitch(t *testing.T, tests []lookupOrderTest) {
	origGetHostname := getHostname
	defer func() { getHostname = origGetHostname }()
	defer setSystemNSS(getSystemNSS(), 0)

	dnsConf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer dnsConf.teardown()

	for _, tt := range tests {
		if !dnsConf.forceUpdateConf(tt.resolv, time.Now().Add(time.Hour)) {
			t.Errorf("%s: failed to change resolv config", tt.name)
			continue
		}

		getHostname = func() (string, error) { return tt.localhost, nil }
		setSystemNSS(tt.nss, time.Hour)

		var GOOSs = []string{"aix", "darwin", "dragonfly", "freebsd", "hurd",
			"illumos", "ios", "linux", "netbsd", "openbsd",
			"solaris", "plan9", "android", "windows", "zos"}

		if !slices.Contains(GOOSs, runtime.GOOS) {
			GOOSs = append(GOOSs, runtime.GOOS)
		}

		for _, os := range GOOSs {
			expect := tt.expect
			if expect != nil {
				// All GOOS in this block of code don't use resolverDynamic.
				// Keep this in sync with conf.go initConfVal.

				if os == "ios" || os == "darwin" {
					expect = maps.Clone(expect)
					delete(expect, resolverDynamic)
				}

				// Plan9, android, windows don't care about
				// the possible system configuration or any runtime informations,
				// so run all tests that where designed for other platforms,
				// but just assert that it always returns the same order.
				if os == "plan9" || os == "android" {
					expect = map[resolverType]hostLookupOrder{
						resolverCgo: hostLookupCgo,
						resolverGo:  hostLookupFilesDNS,
					}
				}

				if os == "windows" {
					expect = map[resolverType]hostLookupOrder{
						resolverCgo: hostLookupCgo,
						resolverGo:  hostLookupDNS,
					}
				}
			}

			if tt.expectGOOS[os] != nil {
				expect = tt.expectGOOS[os]
			}

			var localhost = tt.localhost
			if localhost == "" {
				localhost = "myhostname"
			}

			for resolver, want := range expect {
				c := &conf{resolver: resolver, hasMDNSAllow: tt.mdnsAllow, goos: os}
				gotOrder, _ := c.hostLookupOrder(tt.resolver, tt.host)
				if gotOrder != want {
					t.Errorf(
						"GOOS: %s\n\t%s\n\tresolver: %v\n\tsystem hostname: '%v'\n\thostLookupOrder(%q) = %v; want %v",
						os, tt.name, resolver, localhost, tt.host, gotOrder, want,
					)
				}
			}
		}
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

var testDefaultResolvConf = &dnsConfig{
	servers:  defaultNS,
	ndots:    1,
	timeout:  5,
	attempts: 2,
}

func TestConfHostLookupOrderNsswitch(t *testing.T) {
	nsswithTests := []struct {
		name                 string
		nss                  any // string (to be parsed) or directly *nssConf.
		orderResovlerDynamic hostLookupOrder
		orderResovlerGo      hostLookupOrder

		// solaris has different handing for emtpy (hosts) and non-existent nsswitch.conf.
		solarisResolverDynamic hostLookupOrder
	}{
		{"", "hosts: files dns", hostLookupFilesDNS, hostLookupFilesDNS, hostLookupFilesDNS},
		{"", "hosts: dns files", hostLookupDNSFiles, hostLookupDNSFiles, hostLookupDNSFiles},
		{"", "hosts: dns", hostLookupDNS, hostLookupDNS, hostLookupDNS},
		{"", "hosts: files", hostLookupFiles, hostLookupFiles, hostLookupFiles},
		{"", "hosts: unknown files dns", hostLookupCgo, hostLookupFilesDNS, hostLookupCgo},
		{"", "hosts: dns foo files", hostLookupCgo, hostLookupFilesDNS, hostLookupCgo},
		{"", "hosts: files bar dns", hostLookupCgo, hostLookupFilesDNS, hostLookupCgo},

		// List all the default criteria for dns, but then has a
		// non-standard but redundant notfound=return for the files.
		{"", "hosts: dns [success=return notfound=continue unavail=continue tryagain=continue] files [notfound=return]",
			hostLookupDNSFiles, hostLookupDNSFiles, hostLookupDNSFiles},

		// Common configurations on desptop systems with systemd-resolved.
		{"", "hosts: mymachines resolve [!UNAVAIL=return] files myhostname dns", hostLookupCgo, hostLookupFilesDNS, hostLookupCgo},
		{"", "hosts: resolve [!UNAVAIL=return] files myhostname dns", hostLookupCgo, hostLookupFilesDNS, hostLookupCgo},
		{"", "hosts: resolve [!UNAVAIL=return] files dns", hostLookupCgo, hostLookupFilesDNS, hostLookupCgo},

		{"", "foo:", hostLookupFilesDNS, hostLookupFilesDNS, hostLookupCgo},
		{"", "hosts:", hostLookupFilesDNS, hostLookupFilesDNS, hostLookupCgo},
		{"", "hosts:   ", hostLookupFilesDNS, hostLookupFilesDNS, hostLookupCgo},

		{"empty nsswitch hosts", &nssConf{}, hostLookupFilesDNS, hostLookupFilesDNS, hostLookupCgo},
		{"file does not exist", &nssConf{err: fs.ErrNotExist}, hostLookupFilesDNS, hostLookupFilesDNS, hostLookupCgo},
		{"error while parsing", &nssConf{err: errors.New("parsing error")}, hostLookupCgo, hostLookupFilesDNS, hostLookupCgo},
	}

	var tests []lookupOrderTest

	for _, tt := range nsswithTests {
		var nss *nssConf
		switch v := tt.nss.(type) {
		case string:
			nss = nssStr(t, v)
		case *nssConf:
			nss = v
		default:
			panic("unreachable")
		}

		c := lookupOrderTest{
			name:   fmt.Sprintf("nsswitch: '%v', resolv.conf: default", tt.nss),
			resolv: testDefaultResolvConf,
			nss:    nss,
			host:   "google.com",

			expect: map[resolverType]hostLookupOrder{
				resolverDynamic: tt.orderResovlerDynamic,
				resolverCgo:     hostLookupCgo,
				resolverGo:      tt.orderResovlerGo,
			},

			expectGOOS: map[string]map[resolverType]hostLookupOrder{
				"solaris": map[resolverType]hostLookupOrder{
					resolverDynamic: tt.solarisResolverDynamic,
					resolverCgo:     hostLookupCgo,
					resolverGo:      tt.orderResovlerGo,
				},

				// Ignore this test on obenbsd goos.
				// Openbsd doesn't use nsswitch.
				"openbsd": make(map[resolverType]hostLookupOrder),
			},
		}

		if tt.name != "" {
			c.name = fmt.Sprintf("nsswitch: '%v', resolv.conf: default", tt.name)
		}

		tests = append(tests, c)
	}

	testConfHostLookupOrderNsswitch(t, tests)
}

func TestConfHostLookupOrderResolvConf(t *testing.T) {
	var testsResolv = []struct {
		name                 string
		resolv               *dnsConfig
		nss                  string
		orderResovlerDynamic hostLookupOrder
		orderResovlerGo      hostLookupOrder
	}{
		{"default", testDefaultResolvConf, "hosts: dns files", hostLookupDNSFiles, hostLookupDNSFiles},
		{"default", testDefaultResolvConf, "hosts: files", hostLookupFiles, hostLookupFiles},

		{"file not found", &dnsConfig{err: fs.ErrNotExist}, "hosts: dns files", hostLookupDNSFiles, hostLookupDNSFiles},
		{"file not found", &dnsConfig{err: fs.ErrNotExist}, "hosts: files", hostLookupFiles, hostLookupFiles},

		{"permission error", &dnsConfig{err: fs.ErrPermission}, "hosts: dns files", hostLookupDNSFiles, hostLookupDNSFiles},
		{"permission error", &dnsConfig{err: fs.ErrPermission}, "hosts: files", hostLookupFiles, hostLookupFiles},

		{"unknownOpt", &dnsConfig{unknownOpt: true}, "hosts: dns files", hostLookupCgo, hostLookupFilesDNS},
		{"unknownOpt", &dnsConfig{unknownOpt: true}, "hosts: files", hostLookupCgo, hostLookupFilesDNS},
		{"parsing error", &dnsConfig{err: errors.New("parsing error")}, "hosts: dns files", hostLookupCgo, hostLookupFilesDNS},
		{"parsing error", &dnsConfig{err: errors.New("parsing error")}, "hosts: files", hostLookupCgo, hostLookupFilesDNS},

		{"default", testDefaultResolvConf, "hosts: dns unknown files", hostLookupCgo, hostLookupFilesDNS},
		{"file not found", &dnsConfig{err: fs.ErrNotExist}, "hosts: dns unknown files", hostLookupCgo, hostLookupFilesDNS},
		{"permission error", &dnsConfig{err: fs.ErrPermission}, "hosts: dns unknown files", hostLookupCgo, hostLookupFilesDNS},
		{"unknownOpt", &dnsConfig{unknownOpt: true}, "hosts: dns unknown files", hostLookupCgo, hostLookupFilesDNS},
		{"parsing error", &dnsConfig{err: errors.New("parsing error")}, "hosts: dns unknown files", hostLookupCgo, hostLookupFilesDNS},
	}

	var tests []lookupOrderTest

	for _, tt := range testsResolv {
		c := lookupOrderTest{
			name:   fmt.Sprintf("resolv.conf: '%v', nsswitch.conf: '%v'", tt.name, tt.nss),
			resolv: tt.resolv,
			nss:    nssStr(t, tt.nss),
			host:   "google.com",

			expect: map[resolverType]hostLookupOrder{
				resolverDynamic: tt.orderResovlerDynamic,
				resolverCgo:     hostLookupCgo,
				resolverGo:      tt.orderResovlerGo,
			},
			expectGOOS: map[string]map[resolverType]hostLookupOrder{
				// Openbsd is tested in separate test (below).
				"openbsd": make(map[resolverType]hostLookupOrder),
			},
		}

		tests = append(tests, c)
	}

	testConfHostLookupOrderNsswitch(t, tests)
}

func TestConfHostLookupOrderResolvConfOpenbsd(t *testing.T) {
	var testCases = []struct {
		name   string
		resolv *dnsConfig

		orderResovlerDynamic hostLookupOrder
		orderResovlerGo      hostLookupOrder
	}{
		{"lookup: bind file", &dnsConfig{lookup: []string{"bind", "file"}}, hostLookupDNSFiles, hostLookupDNSFiles},
		{"lookup: file bind", &dnsConfig{lookup: []string{"file", "bind"}}, hostLookupFilesDNS, hostLookupFilesDNS},
		{"lookup: bind", &dnsConfig{lookup: []string{"bind"}}, hostLookupDNS, hostLookupDNS},
		{"lookup: file", &dnsConfig{lookup: []string{"file"}}, hostLookupFiles, hostLookupFiles},

		{"lookup empty", testDefaultResolvConf, hostLookupDNSFiles, hostLookupDNSFiles},

		{"file not exist", &dnsConfig{err: fs.ErrNotExist}, hostLookupFiles, hostLookupFiles},
		{"permission error", &dnsConfig{err: fs.ErrPermission}, hostLookupCgo, hostLookupFilesDNS},
		{"parsing error", &dnsConfig{err: errors.New("parsing error")}, hostLookupCgo, hostLookupFilesDNS},

		{"unknown opt", &dnsConfig{unknownOpt: true}, hostLookupCgo, hostLookupFilesDNS},
		{"lookup: bind unknown file", &dnsConfig{lookup: []string{"bind", "unknown", "file"}}, hostLookupCgo, hostLookupFilesDNS},
	}

	var tests []lookupOrderTest

	for _, tt := range testCases {
		c := lookupOrderTest{
			name:   fmt.Sprintf("resolv.conf: '%v'", tt.name),
			resolv: tt.resolv,
			nss:    nil, // not used by openbsd
			host:   "google.com",

			expectGOOS: map[string]map[resolverType]hostLookupOrder{
				"openbsd": map[resolverType]hostLookupOrder{
					resolverDynamic: tt.orderResovlerDynamic,
					resolverCgo:     hostLookupCgo,
					resolverGo:      tt.orderResovlerGo,
				},
			},
		}

		tests = append(tests, c)
	}

	testConfHostLookupOrderNsswitch(t, tests)
}

func TestConfHostLookupOrderNssMyHostname(t *testing.T) {
	var myhostnameTests = []struct {
		host         string
		localhost    string
		orderDynamic hostLookupOrder
	}{
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
		{"", "myhostname", hostLookupFilesDNS}, // Issue 13623

	}

	var tests []lookupOrderTest

	for _, tt := range myhostnameTests {
		tests = append(tests, lookupOrderTest{
			name:      "nsswitch: hosts: files myhostname dns, default resolv.conf",
			resolv:    testDefaultResolvConf,
			host:      tt.host,
			localhost: tt.localhost,
			nss:       nssStr(t, "hosts: files myhostname dns"),
			expect: map[resolverType]hostLookupOrder{
				resolverDynamic: tt.orderDynamic,
				resolverCgo:     hostLookupCgo,
				resolverGo:      hostLookupFilesDNS,
			},
			expectGOOS: map[string]map[resolverType]hostLookupOrder{
				"openbsd": make(map[resolverType]hostLookupOrder),
			},
		})
	}

	testConfHostLookupOrderNsswitch(t, tests)
}

func TestConfHostLookupOrderSpecialHostnames(t *testing.T) {
	var tests = []lookupOrderTest{
		{
			name:   "special hostname: x\\.com",
			resolv: testDefaultResolvConf,
			nss:    nssStr(t, "hosts: files dns"),
			host:   "x\\.com", // punt on weird glibc escape

			expect: map[resolverType]hostLookupOrder{
				resolverDynamic: hostLookupCgo,
				resolverCgo:     hostLookupCgo,
				resolverGo:      hostLookupFilesDNS,
			},
		},
		{
			name:   "special hostname: go.dev%en0",
			resolv: testDefaultResolvConf,
			nss:    nssStr(t, "hosts: files dns"),
			host:   "go.dev%en0", //  IPv6 zones

			expect: map[resolverType]hostLookupOrder{
				resolverDynamic: hostLookupCgo,
				resolverCgo:     hostLookupCgo,
				resolverGo:      hostLookupFilesDNS,
			},
		},
	}

	testConfHostLookupOrderNsswitch(t, tests)
}

func TestConfHostLookupOrderMDNS(t *testing.T) {
	const nsswitchUbuntuTrustyAvahi = "hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4"
	const nsswitchUbuntu14 = "hosts: files myhostname mdns4_minimal [NOTFOUND=return] dns mdns4"

	var mdsnTests = []struct {
		nss string

		host         string
		localhost    string
		hasMdnsAllow bool

		orderDynamic hostLookupOrder
		orderGo      hostLookupOrder
	}{
		// nsswitch.conf from ubuntu trusty with avahi
		{nsswitchUbuntuTrustyAvahi, "foo.local", "", false, hostLookupCgo, hostLookupFilesDNS},
		{nsswitchUbuntuTrustyAvahi, "foo.LOCAL", "", false, hostLookupCgo, hostLookupFilesDNS},
		{nsswitchUbuntuTrustyAvahi, "foo.local.", "", false, hostLookupCgo, hostLookupFilesDNS},
		{nsswitchUbuntuTrustyAvahi, "foo.LOCAL", "", false, hostLookupCgo, hostLookupFilesDNS},
		{nsswitchUbuntuTrustyAvahi, "google.com", "", false, hostLookupFilesDNS, hostLookupFilesDNS},

		// nsswitch from ubuntu14.04.02
		{nsswitchUbuntu14, "foo.local", "hostname", false, hostLookupCgo, hostLookupFilesDNS},
		{nsswitchUbuntu14, "hostname", "hostname", false, hostLookupCgo, hostLookupFilesDNS},
		{nsswitchUbuntu14, "somehostname", "hostname", false, hostLookupFilesDNS, hostLookupFilesDNS},
		{nsswitchUbuntu14, "google.com", "hostname", false, hostLookupFilesDNS, hostLookupFilesDNS},

		{"hosts: files mdns dns", "foo.local", "", true, hostLookupCgo, hostLookupFilesDNS},
		{"hosts: files mdns dns", "google.com", "", true, hostLookupCgo, hostLookupFilesDNS},
		{"hosts: files mdns dns", "google.com", "", false, hostLookupFilesDNS, hostLookupFilesDNS},

		// TODO: what is the point of fallbacking to hostLookupCgo here? The system doesn't use
		// mdns, so go resolver will do fine here.
		{"hosts: files dns", "foo.local", "", false, hostLookupCgo, hostLookupFilesDNS},
		{"hosts: files mdns dns", "foo.local", "", false, hostLookupCgo, hostLookupFilesDNS},
	}

	var tests []lookupOrderTest
	for _, tt := range mdsnTests {
		tests = append(tests, lookupOrderTest{
			name:      fmt.Sprintf("nsswitch: '%v', default resolv.conf", tt.nss),
			resolver:  nil,
			resolv:    testDefaultResolvConf,
			nss:       nssStr(t, tt.nss),
			host:      tt.host,
			localhost: tt.localhost,
			mdnsAllow: tt.hasMdnsAllow,

			expect: map[resolverType]hostLookupOrder{
				resolverDynamic: tt.orderDynamic,
				resolverCgo:     hostLookupCgo,
				resolverGo:      tt.orderGo,
			},
			expectGOOS: map[string]map[resolverType]hostLookupOrder{
				"openbsd": make(map[resolverType]hostLookupOrder),
			},
		})
	}

	testConfHostLookupOrderNsswitch(t, tests)
}

func TestConfHostLookupOrderPreferGo(t *testing.T) {
	// Issue 24393: make sure "Resolver.PreferGo = true" acts like netgo.
	testConfHostLookupOrderNsswitch(t, []lookupOrderTest{
		{
			resolver: &Resolver{PreferGo: true},
			resolv:   &dnsConfig{lookup: []string{"file"}}, // for openbsd
			nss:      nssStr(t, "hosts: files"),
			host:     "google.com",

			expect: map[resolverType]hostLookupOrder{
				resolverDynamic: hostLookupFiles,
				resolverCgo:     hostLookupFiles,
				resolverGo:      hostLookupFiles,
			},
			expectGOOS: map[string]map[resolverType]hostLookupOrder{
				"android": map[resolverType]hostLookupOrder{
					resolverCgo: hostLookupFilesDNS,
					resolverGo:  hostLookupFilesDNS,
				},
				"plan9": map[resolverType]hostLookupOrder{
					resolverCgo: hostLookupFilesDNS,
					resolverGo:  hostLookupFilesDNS,
				},
				"windows": map[resolverType]hostLookupOrder{
					resolverCgo: hostLookupDNS,
					resolverGo:  hostLookupDNS,
				},
			},
		},
		{
			resolver: &Resolver{PreferGo: true},
			resolv: &dnsConfig{
				lookup: []string{"file", "unknown", "dns", "unknown"}, // for openbsd
			},
			nss:  nssStr(t, "hosts: files unknown dns unknown"),
			host: "google.com",

			expect: map[resolverType]hostLookupOrder{
				resolverDynamic: hostLookupFilesDNS,
				resolverCgo:     hostLookupFilesDNS,
				resolverGo:      hostLookupFilesDNS,
			},
			expectGOOS: map[string]map[resolverType]hostLookupOrder{
				"android": map[resolverType]hostLookupOrder{
					resolverCgo: hostLookupFilesDNS,
					resolverGo:  hostLookupFilesDNS,
				},
				"plan9": map[resolverType]hostLookupOrder{
					resolverCgo: hostLookupFilesDNS,
					resolverGo:  hostLookupFilesDNS,
				},
				"windows": map[resolverType]hostLookupOrder{
					resolverCgo: hostLookupDNS,
					resolverGo:  hostLookupDNS,
				},
			},
		},
	})
}

func TestSystemConf(t *testing.T) {
	systemConf()
}
