// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"reflect"
	"testing"
)

var dnsReadConfigTests = []struct {
	name string
	conf dnsConfig
}{
	{
		name: "testdata/resolv.conf",
		conf: dnsConfig{
			servers:  []string{"8.8.8.8", "2001:4860:4860::8888", "fe80::1%lo0"},
			search:   []string{"localdomain"},
			ndots:    5,
			timeout:  10,
			attempts: 3,
			rotate:   true,
		},
	},
	{
		name: "testdata/domain-resolv.conf",
		conf: dnsConfig{
			servers:  []string{"8.8.8.8"},
			search:   []string{"localdomain"},
			ndots:    1,
			timeout:  5,
			attempts: 2,
		},
	},
	{
		name: "testdata/search-resolv.conf",
		conf: dnsConfig{
			servers:  []string{"8.8.8.8"},
			search:   []string{"test", "invalid"},
			ndots:    1,
			timeout:  5,
			attempts: 2,
		},
	},
	{
		name: "testdata/empty-resolv.conf",
		conf: dnsConfig{
			ndots:    1,
			timeout:  5,
			attempts: 2,
		},
	},
}

func TestDNSReadConfig(t *testing.T) {
	for _, tt := range dnsReadConfigTests {
		conf, err := dnsReadConfig(tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(conf, &tt.conf) {
			t.Errorf("got %v; want %v", conf, &tt.conf)
		}
	}
}
