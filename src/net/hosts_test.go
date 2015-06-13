// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
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
		"testdata/ipv4-hosts", // see golang.org/issue/8996
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
}

func TestLookupStaticHost(t *testing.T) {
	defer func(orig string) { testHookHostsPath = orig }(testHookHostsPath)

	for _, tt := range lookupStaticHostTests {
		testHookHostsPath = tt.name
		for _, ent := range tt.ents {
			addrs := lookupStaticHost(ent.in)
			if !reflect.DeepEqual(addrs, ent.out) {
				t.Errorf("%s, lookupStaticHost(%s) = %v; want %v", tt.name, ent.in, addrs, ent.out)
			}
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
		"testdata/ipv4-hosts", // see golang.org/issue/8996
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
}

func TestLookupStaticAddr(t *testing.T) {
	defer func(orig string) { testHookHostsPath = orig }(testHookHostsPath)

	for _, tt := range lookupStaticAddrTests {
		testHookHostsPath = tt.name
		for _, ent := range tt.ents {
			hosts := lookupStaticAddr(ent.in)
			if !reflect.DeepEqual(hosts, ent.out) {
				t.Errorf("%s, lookupStaticAddr(%s) = %v; want %v", tt.name, ent.in, hosts, ent.out)
			}
		}
	}
}
