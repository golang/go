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
	var char63 = ""
	for i := 0; i < 63; i++ {
		char63 += "a"
	}
	char64 := char63 + "a"
	longDomain := strings.Repeat(char63+".", 5) + "example"

	for _, tc := range dnsNameTests {
		ch <- tc
	}

	ch <- dnsNameTest{char63 + ".com", true}
	ch <- dnsNameTest{char64 + ".com", false}

	// Remember: wire format is two octets longer than presentation
	// (length octets for the first and [root] last labels).
	// 253 is fine:
	ch <- dnsNameTest{longDomain[len(longDomain)-253:], true}
	// A terminal dot doesn't contribute to length:
	ch <- dnsNameTest{longDomain[len(longDomain)-253:] + ".", true}
	// 254 is bad:
	ch <- dnsNameTest{longDomain[len(longDomain)-254:], false}
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
