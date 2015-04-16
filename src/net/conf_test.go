// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"os"
	"strings"
	"testing"
)

type nssHostTest struct {
	host string
	want hostLookupOrder
}

func nssStr(s string) *nssConf { return parseNSSConf(strings.NewReader(s)) }

func TestConfHostLookupOrder(t *testing.T) {
	tests := []struct {
		name      string
		c         *conf
		goos      string
		hostTests []nssHostTest
	}{
		{
			name: "force",
			c: &conf{
				forceCgoLookupHost: true,
				nss:                nssStr("foo: bar"),
			},
			hostTests: []nssHostTest{
				{"foo.local", hostLookupCgo},
				{"google.com", hostLookupCgo},
			},
		},
		{
			name: "ubuntu_trusty_avahi",
			c: &conf{
				nss: nssStr("hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4"),
			},
			hostTests: []nssHostTest{
				{"foo.local", hostLookupCgo},
				{"foo.local.", hostLookupCgo},
				{"foo.LOCAL", hostLookupCgo},
				{"foo.LOCAL.", hostLookupCgo},
				{"google.com", hostLookupFilesDNS},
			},
		},
		{
			name: "freebsdlinux_no_resolv_conf",
			c: &conf{
				goos: "freebsd",
				nss:  nssStr("foo: bar"),
			},
			hostTests: []nssHostTest{{"google.com", hostLookupFilesDNS}},
		},
		// On OpenBSD, no resolv.conf means no DNS.
		{
			name: "openbsd_no_resolv_conf",
			c: &conf{
				goos: "openbsd",
			},
			hostTests: []nssHostTest{{"google.com", hostLookupFiles}},
		},
		{
			name: "solaris_no_nsswitch",
			c: &conf{
				goos: "solaris",
				nss:  &nssConf{err: os.ErrNotExist},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupCgo}},
		},
		{
			name: "openbsd_lookup_bind_file",
			c: &conf{
				goos:   "openbsd",
				resolv: &dnsConfig{lookup: []string{"bind", "file"}},
			},
			hostTests: []nssHostTest{
				{"google.com", hostLookupDNSFiles},
				{"foo.local", hostLookupDNSFiles},
			},
		},
		{
			name: "openbsd_lookup_file_bind",
			c: &conf{
				goos:   "openbsd",
				resolv: &dnsConfig{lookup: []string{"file", "bind"}},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupFilesDNS}},
		},
		{
			name: "openbsd_lookup_bind",
			c: &conf{
				goos:   "openbsd",
				resolv: &dnsConfig{lookup: []string{"bind"}},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupDNS}},
		},
		{
			name: "openbsd_lookup_file",
			c: &conf{
				goos:   "openbsd",
				resolv: &dnsConfig{lookup: []string{"file"}},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupFiles}},
		},
		{
			name: "openbsd_lookup_yp",
			c: &conf{
				goos:   "openbsd",
				resolv: &dnsConfig{lookup: []string{"file", "bind", "yp"}},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupCgo}},
		},
		{
			name: "openbsd_lookup_two",
			c: &conf{
				goos:   "openbsd",
				resolv: &dnsConfig{lookup: []string{"file", "foo"}},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupCgo}},
		},
		{
			name: "openbsd_lookup_empty",
			c: &conf{
				goos:   "openbsd",
				resolv: &dnsConfig{lookup: nil},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupDNSFiles}},
		},
		// glibc lacking an nsswitch.conf, per
		// http://www.gnu.org/software/libc/manual/html_node/Notes-on-NSS-Configuration-File.html
		{
			name: "linux_no_nsswitch.conf",
			c: &conf{
				goos: "linux",
				nss:  &nssConf{err: os.ErrNotExist},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupDNSFiles}},
		},
		{
			name: "files_mdns_dns",
			c:    &conf{nss: nssStr("hosts: files mdns dns")},
			hostTests: []nssHostTest{
				{"x.com", hostLookupFilesDNS},
				{"x.local", hostLookupCgo},
			},
		},
		{
			name: "dns_special_hostnames",
			c:    &conf{nss: nssStr("hosts: dns")},
			hostTests: []nssHostTest{
				{"x.com", hostLookupDNS},
				{"x\\.com", hostLookupCgo},     // punt on weird glibc escape
				{"foo.com%en0", hostLookupCgo}, // and IPv6 zones
			},
		},
		{
			name: "mdns_allow",
			c: &conf{
				nss:          nssStr("hosts: files mdns dns"),
				hasMDNSAllow: true,
			},
			hostTests: []nssHostTest{
				{"x.com", hostLookupCgo},
				{"x.local", hostLookupCgo},
			},
		},
		{
			name: "files_dns",
			c:    &conf{nss: nssStr("hosts: files dns")},
			hostTests: []nssHostTest{
				{"x.com", hostLookupFilesDNS},
				{"x", hostLookupFilesDNS},
				{"x.local", hostLookupCgo},
			},
		},
		{
			name: "dns_files",
			c:    &conf{nss: nssStr("hosts: dns files")},
			hostTests: []nssHostTest{
				{"x.com", hostLookupDNSFiles},
				{"x", hostLookupDNSFiles},
				{"x.local", hostLookupCgo},
			},
		},
		{
			name: "something_custom",
			c:    &conf{nss: nssStr("hosts: dns files something_custom")},
			hostTests: []nssHostTest{
				{"x.com", hostLookupCgo},
			},
		},
		{
			name: "myhostname",
			c:    &conf{nss: nssStr("hosts: files dns myhostname")},
			hostTests: []nssHostTest{
				{"x.com", hostLookupFilesDNS},
				{"somehostname", hostLookupCgo},
			},
		},
		{
			name: "ubuntu14.04.02",
			c:    &conf{nss: nssStr("hosts: files myhostname mdns4_minimal [NOTFOUND=return] dns mdns4")},
			hostTests: []nssHostTest{
				{"x.com", hostLookupFilesDNS},
				{"somehostname", hostLookupCgo},
			},
		},
		// Debian Squeeze is just "dns,files", but lists all
		// the default criteria for dns, but then has a
		// non-standard but redundant notfound=return for the
		// files.
		{
			name: "debian_squeeze",
			c:    &conf{nss: nssStr("hosts: dns [success=return notfound=continue unavail=continue tryagain=continue] files [notfound=return]")},
			hostTests: []nssHostTest{
				{"x.com", hostLookupDNSFiles},
				{"somehostname", hostLookupDNSFiles},
			},
		},
		{
			name: "resolv.conf-unknown",
			c: &conf{
				nss:    nssStr("foo: bar"),
				resolv: &dnsConfig{unknownOpt: true},
			},
			hostTests: []nssHostTest{{"google.com", hostLookupCgo}},
		},
	}
	for _, tt := range tests {
		for _, ht := range tt.hostTests {
			gotOrder := tt.c.hostLookupOrder(ht.host)
			if gotOrder != ht.want {
				t.Errorf("%s: useCgoLookupHost(%q) = %v; want %v", tt.name, ht.host, gotOrder, ht.want)
			}
		}
	}

}
