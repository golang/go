// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"strings"
	"testing"
)

type staticHostEntry struct {
	in  string
	out []string
}

var lookupStaticHostTests = []struct {
	name string
	ents []staticHostEntry
}{
	{
		"testdata/hosts",
		[]staticHostEntry{
			{"odin", []string{"127.0.0.2", "127.0.0.3", "::2"}},
			{"thor", []string{"127.1.1.1"}},
			{"ullr", []string{"127.1.1.2"}},
			{"ullrhost", []string{"127.1.1.2"}},
			{"localhost", []string{"fe80::1%lo0"}},
		},
	},
	{
		"testdata/singleline-hosts", // see golang.org/issue/6646
		[]staticHostEntry{
			{"odin", []string{"127.0.0.2"}},
		},
	},
	{
		"testdata/ipv4-hosts",
		[]staticHostEntry{
			{"localhost", []string{"127.0.0.1", "127.0.0.2", "127.0.0.3"}},
			{"localhost.localdomain", []string{"127.0.0.3"}},
		},
	},
	{
		"testdata/ipv6-hosts", // see golang.org/issue/8996
		[]staticHostEntry{
			{"localhost", []string{"::1", "fe80::1", "fe80::2%lo0", "fe80::3%lo0"}},
			{"localhost.localdomain", []string{"fe80::3%lo0"}},
		},
	},
	{
		"testdata/case-hosts", // see golang.org/issue/12806
		[]staticHostEntry{
			{"PreserveMe", []string{"127.0.0.1", "::1"}},
			{"PreserveMe.local", []string{"127.0.0.1", "::1"}},
		},
	},
}

func TestLookupStaticHost(t *testing.T) {
	defer func(orig string) { testHookHostsPath = orig }(testHookHostsPath)

	for _, tt := range lookupStaticHostTests {
		testHookHostsPath = tt.name
		for _, ent := range tt.ents {
			testStaticHost(t, tt.name, ent)
		}
	}
}

func testStaticHost(t *testing.T, hostsPath string, ent staticHostEntry) {
	ins := []string{ent.in, absDomainName([]byte(ent.in)), strings.ToLower(ent.in), strings.ToUpper(ent.in)}
	for _, in := range ins {
		addrs := lookupStaticHost(in)
		if !reflect.DeepEqual(addrs, ent.out) {
			t.Errorf("%s, lookupStaticHost(%s) = %v; want %v", hostsPath, in, addrs, ent.out)
		}
	}
}

var lookupStaticAddrTests = []struct {
	name string
	ents []staticHostEntry
}{
	{
		"testdata/hosts",
		[]staticHostEntry{
			{"255.255.255.255", []string{"broadcasthost"}},
			{"127.0.0.2", []string{"odin"}},
			{"127.0.0.3", []string{"odin"}},
			{"::2", []string{"odin"}},
			{"127.1.1.1", []string{"thor"}},
			{"127.1.1.2", []string{"ullr", "ullrhost"}},
			{"fe80::1%lo0", []string{"localhost"}},
		},
	},
	{
		"testdata/singleline-hosts", // see golang.org/issue/6646
		[]staticHostEntry{
			{"127.0.0.2", []string{"odin"}},
		},
	},
	{
		"testdata/ipv4-hosts",
		[]staticHostEntry{
			{"127.0.0.1", []string{"localhost"}},
			{"127.0.0.2", []string{"localhost"}},
			{"127.0.0.3", []string{"localhost", "localhost.localdomain"}},
		},
	},
	{
		"testdata/ipv6-hosts", // see golang.org/issue/8996
		[]staticHostEntry{
			{"::1", []string{"localhost"}},
			{"fe80::1", []string{"localhost"}},
			{"fe80::2%lo0", []string{"localhost"}},
			{"fe80::3%lo0", []string{"localhost", "localhost.localdomain"}},
		},
	},
	{
		"testdata/case-hosts", // see golang.org/issue/12806
		[]staticHostEntry{
			{"127.0.0.1", []string{"PreserveMe", "PreserveMe.local"}},
			{"::1", []string{"PreserveMe", "PreserveMe.local"}},
		},
	},
}

func TestLookupStaticAddr(t *testing.T) {
	defer func(orig string) { testHookHostsPath = orig }(testHookHostsPath)

	for _, tt := range lookupStaticAddrTests {
		testHookHostsPath = tt.name
		for _, ent := range tt.ents {
			testStaticAddr(t, tt.name, ent)
		}
	}
}

func testStaticAddr(t *testing.T, hostsPath string, ent staticHostEntry) {
	hosts := lookupStaticAddr(ent.in)
	for i := range ent.out {
		ent.out[i] = absDomainName([]byte(ent.out[i]))
	}
	if !reflect.DeepEqual(hosts, ent.out) {
		t.Errorf("%s, lookupStaticAddr(%s) = %v; want %v", hostsPath, ent.in, hosts, ent.out)
	}
}

func TestHostCacheModification(t *testing.T) {
	// Ensure that programs can't modify the internals of the host cache.
	// See https://golang.org/issues/14212.
	defer func(orig string) { testHookHostsPath = orig }(testHookHostsPath)

	testHookHostsPath = "testdata/ipv4-hosts"
	ent := staticHostEntry{"localhost", []string{"127.0.0.1", "127.0.0.2", "127.0.0.3"}}
	testStaticHost(t, testHookHostsPath, ent)
	// Modify the addresses return by lookupStaticHost.
	addrs := lookupStaticHost(ent.in)
	for i := range addrs {
		addrs[i] += "junk"
	}
	testStaticHost(t, testHookHostsPath, ent)

	testHookHostsPath = "testdata/ipv6-hosts"
	ent = staticHostEntry{"::1", []string{"localhost"}}
	testStaticAddr(t, testHookHostsPath, ent)
	// Modify the hosts return by lookupStaticAddr.
	hosts := lookupStaticAddr(ent.in)
	for i := range hosts {
		hosts[i] += "junk"
	}
	testStaticAddr(t, testHookHostsPath, ent)
}
