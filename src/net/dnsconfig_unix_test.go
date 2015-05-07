// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"os"
	"reflect"
	"testing"
)

var dnsReadConfigTests = []struct {
	name string
	want *dnsConfig
}{
	{
		name: "testdata/resolv.conf",
		want: &dnsConfig{
			servers:    []string{"8.8.8.8", "2001:4860:4860::8888", "fe80::1%lo0"},
			search:     []string{"localdomain"},
			ndots:      5,
			timeout:    10,
			attempts:   3,
			rotate:     true,
			unknownOpt: true, // the "options attempts 3" line
		},
	},
	{
		name: "testdata/domain-resolv.conf",
		want: &dnsConfig{
			servers:  []string{"8.8.8.8"},
			search:   []string{"localdomain"},
			ndots:    1,
			timeout:  5,
			attempts: 2,
		},
	},
	{
		name: "testdata/search-resolv.conf",
		want: &dnsConfig{
			servers:  []string{"8.8.8.8"},
			search:   []string{"test", "invalid"},
			ndots:    1,
			timeout:  5,
			attempts: 2,
		},
	},
	{
		name: "testdata/empty-resolv.conf",
		want: &dnsConfig{
			servers:  defaultNS,
			ndots:    1,
			timeout:  5,
			attempts: 2,
		},
	},
	{
		name: "testdata/openbsd-resolv.conf",
		want: &dnsConfig{
			ndots:    1,
			timeout:  5,
			attempts: 2,
			lookup:   []string{"file", "bind"},
			servers:  []string{"169.254.169.254", "10.240.0.1"},
			search:   []string{"c.symbolic-datum-552.internal."},
		},
	},
}

func TestDNSReadConfig(t *testing.T) {
	for _, tt := range dnsReadConfigTests {
		conf := dnsReadConfig(tt.name)
		if conf.err != nil {
			t.Fatal(conf.err)
		}
		if !reflect.DeepEqual(conf, tt.want) {
			t.Errorf("%s:\ngot: %+v\nwant: %+v", tt.name, conf, tt.want)
		}
	}
}

func TestDNSReadMissingFile(t *testing.T) {
	conf := dnsReadConfig("a-nonexistent-file")
	if !os.IsNotExist(conf.err) {
		t.Errorf("missing resolv.conf:\ngot: %v\nwant: %v", conf.err, os.ErrNotExist)
	}
	conf.err = nil
	want := &dnsConfig{
		servers:  defaultNS,
		ndots:    1,
		timeout:  5,
		attempts: 2,
	}
	if !reflect.DeepEqual(conf, want) {
		t.Errorf("missing resolv.conf:\ngot: %+v\nwant: %+v", conf, want)
	}
}
