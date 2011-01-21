// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"testing"
	"runtime"
)

type testCase struct {
	name   string
	result bool
}

var tests = []testCase{
	// RFC2181, section 11.
	{"_xmpp-server._tcp.google.com", true},
	{"_xmpp-server._tcp.google.com", true},
	{"foo.com", true},
	{"1foo.com", true},
	{"26.0.0.73.com", true},
	{"fo-o.com", true},
	{"fo1o.com", true},
	{"foo1.com", true},
	{"a.b..com", false},
}

func getTestCases(ch chan<- testCase) {
	defer close(ch)
	var char59 = ""
	var char63 = ""
	var char64 = ""
	for i := 0; i < 59; i++ {
		char59 += "a"
	}
	char63 = char59 + "aaaa"
	char64 = char63 + "a"

	for _, tc := range tests {
		ch <- tc
	}

	ch <- testCase{char63 + ".com", true}
	ch <- testCase{char64 + ".com", false}
	// 255 char name is fine:
	ch <- testCase{char59 + "." + char63 + "." + char63 + "." +
		char63 + ".com",
		true}
	// 256 char name is bad:
	ch <- testCase{char59 + "a." + char63 + "." + char63 + "." +
		char63 + ".com",
		false}
}

func TestDNSNames(t *testing.T) {
	if runtime.GOOS == "windows" {
		return
	}
	ch := make(chan testCase)
	go getTestCases(ch)
	for tc := range ch {
		if isDomainName(tc.name) != tc.result {
			t.Errorf("isDomainName(%v) failed: Should be %v",
				tc.name, tc.result)
		}
	}
}
