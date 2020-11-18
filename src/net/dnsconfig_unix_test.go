// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"errors"
	"io/fs"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

var dnsReadConfigTests = []struct {
	name string
	want *dnsConfig
}{
	{
		name: "testdata/resolv.conf",
		want: &dnsConfig{
			servers:    []string{"8.8.8.8:53", "[2001:4860:4860::8888]:53", "[fe80::1%lo0]:53"},
			search:     []string{"localdomain."},
			ndots:      5,
			timeout:    10 * time.Second,
			attempts:   3,
			rotate:     true,
			unknownOpt: true, // the "options attempts 3" line
		},
	},
	{
		name: "testdata/domain-resolv.conf",
		want: &dnsConfig{
			servers:  []string{"8.8.8.8:53"},
			search:   []string{"localdomain."},
			ndots:    1,
			timeout:  5 * time.Second,
			attempts: 2,
		},
	},
	{
		name: "testdata/search-resolv.conf",
		want: &dnsConfig{
			servers:  []string{"8.8.8.8:53"},
			search:   []string{"test.", "invalid."},
			ndots:    1,
			timeout:  5 * time.Second,
			attempts: 2,
		},
	},
	{
		name: "testdata/empty-resolv.conf",
		want: &dnsConfig{
			servers:  defaultNS,
			ndots:    1,
			timeout:  5 * time.Second,
			attempts: 2,
			search:   []string{"domain.local."},
		},
	},
	{
		name: "testdata/invalid-ndots-resolv.conf",
		want: &dnsConfig{
			servers:  defaultNS,
			ndots:    0,
			timeout:  5 * time.Second,
			attempts: 2,
			search:   []string{"domain.local."},
		},
	},
	{
		name: "testdata/large-ndots-resolv.conf",
		want: &dnsConfig{
			servers:  defaultNS,
			ndots:    15,
			timeout:  5 * time.Second,
			attempts: 2,
			search:   []string{"domain.local."},
		},
	},
	{
		name: "testdata/negative-ndots-resolv.conf",
		want: &dnsConfig{
			servers:  defaultNS,
			ndots:    0,
			timeout:  5 * time.Second,
			attempts: 2,
			search:   []string{"domain.local."},
		},
	},
	{
		name: "testdata/openbsd-resolv.conf",
		want: &dnsConfig{
			ndots:    1,
			timeout:  5 * time.Second,
			attempts: 2,
			lookup:   []string{"file", "bind"},
			servers:  []string{"169.254.169.254:53", "10.240.0.1:53"},
			search:   []string{"c.symbolic-datum-552.internal."},
		},
	},
	{
		name: "testdata/single-request-resolv.conf",
		want: &dnsConfig{
			servers:       defaultNS,
			ndots:         1,
			singleRequest: true,
			timeout:       5 * time.Second,
			attempts:      2,
			search:        []string{"domain.local."},
		},
	},
	{
		name: "testdata/single-request-reopen-resolv.conf",
		want: &dnsConfig{
			servers:       defaultNS,
			ndots:         1,
			singleRequest: true,
			timeout:       5 * time.Second,
			attempts:      2,
			search:        []string{"domain.local."},
		},
	},
	{
		name: "testdata/linux-use-vc-resolv.conf",
		want: &dnsConfig{
			servers:  defaultNS,
			ndots:    1,
			useTCP:   true,
			timeout:  5 * time.Second,
			attempts: 2,
			search:   []string{"domain.local."},
		},
	},
	{
		name: "testdata/freebsd-usevc-resolv.conf",
		want: &dnsConfig{
			servers:  defaultNS,
			ndots:    1,
			useTCP:   true,
			timeout:  5 * time.Second,
			attempts: 2,
			search:   []string{"domain.local."},
		},
	},
	{
		name: "testdata/openbsd-tcp-resolv.conf",
		want: &dnsConfig{
			servers:  defaultNS,
			ndots:    1,
			useTCP:   true,
			timeout:  5 * time.Second,
			attempts: 2,
			search:   []string{"domain.local."},
		},
	},
}

func TestDNSReadConfig(t *testing.T) {
	origGetHostname := getHostname
	defer func() { getHostname = origGetHostname }()
	getHostname = func() (string, error) { return "host.domain.local", nil }

	for _, tt := range dnsReadConfigTests {
		conf := dnsReadConfig(tt.name)
		if conf.err != nil {
			t.Fatal(conf.err)
		}
		conf.mtime = time.Time{}
		if !reflect.DeepEqual(conf, tt.want) {
			t.Errorf("%s:\ngot: %+v\nwant: %+v", tt.name, conf, tt.want)
		}
	}
}

func TestDNSReadMissingFile(t *testing.T) {
	origGetHostname := getHostname
	defer func() { getHostname = origGetHostname }()
	getHostname = func() (string, error) { return "host.domain.local", nil }

	conf := dnsReadConfig("a-nonexistent-file")
	if !os.IsNotExist(conf.err) {
		t.Errorf("missing resolv.conf:\ngot: %v\nwant: %v", conf.err, fs.ErrNotExist)
	}
	conf.err = nil
	want := &dnsConfig{
		servers:  defaultNS,
		ndots:    1,
		timeout:  5 * time.Second,
		attempts: 2,
		search:   []string{"domain.local."},
	}
	if !reflect.DeepEqual(conf, want) {
		t.Errorf("missing resolv.conf:\ngot: %+v\nwant: %+v", conf, want)
	}
}

var dnsDefaultSearchTests = []struct {
	name string
	err  error
	want []string
}{
	{
		name: "host.long.domain.local",
		want: []string{"long.domain.local."},
	},
	{
		name: "host.local",
		want: []string{"local."},
	},
	{
		name: "host",
		want: nil,
	},
	{
		name: "host.domain.local",
		err:  errors.New("errored"),
		want: nil,
	},
	{
		// ensures we don't return []string{""}
		// which causes duplicate lookups
		name: "foo.",
		want: nil,
	},
}

func TestDNSDefaultSearch(t *testing.T) {
	origGetHostname := getHostname
	defer func() { getHostname = origGetHostname }()

	for _, tt := range dnsDefaultSearchTests {
		getHostname = func() (string, error) { return tt.name, tt.err }
		got := dnsDefaultSearch()
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("dnsDefaultSearch with hostname %q and error %+v = %q, wanted %q", tt.name, tt.err, got, tt.want)
		}
	}
}

func TestDNSNameLength(t *testing.T) {
	origGetHostname := getHostname
	defer func() { getHostname = origGetHostname }()
	getHostname = func() (string, error) { return "host.domain.local", nil }

	var char63 = ""
	for i := 0; i < 63; i++ {
		char63 += "a"
	}
	longDomain := strings.Repeat(char63+".", 5) + "example"

	for _, tt := range dnsReadConfigTests {
		conf := dnsReadConfig(tt.name)
		if conf.err != nil {
			t.Fatal(conf.err)
		}

		var shortestSuffix int
		for _, suffix := range tt.want.search {
			if shortestSuffix == 0 || len(suffix) < shortestSuffix {
				shortestSuffix = len(suffix)
			}
		}

		// Test a name that will be maximally long when prefixing the shortest
		// suffix (accounting for the intervening dot).
		longName := longDomain[len(longDomain)-254+1+shortestSuffix:]
		if longName[0] == '.' || longName[1] == '.' {
			longName = "aa." + longName[3:]
		}
		for _, fqdn := range conf.nameList(longName) {
			if len(fqdn) > 254 {
				t.Errorf("got %d; want less than or equal to 254", len(fqdn))
			}
		}

		// Now test a name that's too long for suffixing.
		unsuffixable := "a." + longName[1:]
		unsuffixableResults := conf.nameList(unsuffixable)
		if len(unsuffixableResults) != 1 {
			t.Errorf("suffixed names %v; want []", unsuffixableResults[1:])
		}

		// Now test a name that's too long for DNS.
		tooLong := "a." + longDomain
		tooLongResults := conf.nameList(tooLong)
		if tooLongResults != nil {
			t.Errorf("suffixed names %v; want nil", tooLongResults)
		}
	}
}
