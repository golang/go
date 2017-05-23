// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"strings"
	"testing"
)

type dnsNameTest struct {
	name   string
	result bool
}

var dnsNameTests = []dnsNameTest{
	// RFC 2181, section 11.
	{"_xmpp-server._tcp.google.com", true},
	{"foo.com", true},
	{"1foo.com", true},
	{"26.0.0.73.com", true},
	{"fo-o.com", true},
	{"fo1o.com", true},
	{"foo1.com", true},
	{"a.b..com", false},
	{"a.b-.com", false},
	{"a.b.com-", false},
	{"a.b..", false},
	{"b.com.", true},
}

func emitDNSNameTest(ch chan<- dnsNameTest) {
	defer close(ch)
	var char59 = ""
	var char63 = ""
	var char64 = ""
	for i := 0; i < 59; i++ {
		char59 += "a"
	}
	char63 = char59 + "aaaa"
	char64 = char63 + "a"

	for _, tc := range dnsNameTests {
		ch <- tc
	}

	ch <- dnsNameTest{char63 + ".com", true}
	ch <- dnsNameTest{char64 + ".com", false}
	// 255 char name is fine:
	ch <- dnsNameTest{char59 + "." + char63 + "." + char63 + "." +
		char63 + ".com",
		true}
	// 256 char name is bad:
	ch <- dnsNameTest{char59 + "a." + char63 + "." + char63 + "." +
		char63 + ".com",
		false}
}

func TestDNSName(t *testing.T) {
	ch := make(chan dnsNameTest)
	go emitDNSNameTest(ch)
	for tc := range ch {
		if isDomainName(tc.name) != tc.result {
			t.Errorf("isDomainName(%q) = %v; want %v", tc.name, !tc.result, tc.result)
		}
	}
}

func BenchmarkDNSName(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	benchmarks := append(dnsNameTests, []dnsNameTest{
		{strings.Repeat("a", 63), true},
		{strings.Repeat("a", 64), false},
	}...)
	for n := 0; n < b.N; n++ {
		for _, tc := range benchmarks {
			if isDomainName(tc.name) != tc.result {
				b.Errorf("isDomainName(%q) = %v; want %v", tc.name, !tc.result, tc.result)
			}
		}
	}
}
