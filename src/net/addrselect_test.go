// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package net

import (
	"reflect"
	"testing"
)

func TestSortByRFC6724(t *testing.T) {
	tests := []struct {
		in      []IPAddr
		srcs    []IP
		want    []IPAddr
		reverse bool // also test it starting backwards
	}{
		// Examples from RFC 6724 section 10.2:

		// Prefer matching scope.
		{
			in: []IPAddr{
				{IP: ParseIP("2001:db8:1::1")},
				{IP: ParseIP("198.51.100.121")},
			},
			srcs: []IP{
				ParseIP("2001:db8:1::2"),
				ParseIP("169.254.13.78"),
			},
			want: []IPAddr{
				{IP: ParseIP("2001:db8:1::1")},
				{IP: ParseIP("198.51.100.121")},
			},
			reverse: true,
		},

		// Prefer matching scope.
		{
			in: []IPAddr{
				{IP: ParseIP("2001:db8:1::1")},
				{IP: ParseIP("198.51.100.121")},
			},
			srcs: []IP{
				ParseIP("fe80::1"),
				ParseIP("198.51.100.117"),
			},
			want: []IPAddr{
				{IP: ParseIP("198.51.100.121")},
				{IP: ParseIP("2001:db8:1::1")},
			},
			reverse: true,
		},

		// Prefer higher precedence.
		{
			in: []IPAddr{
				{IP: ParseIP("2001:db8:1::1")},
				{IP: ParseIP("10.1.2.3")},
			},
			srcs: []IP{
				ParseIP("2001:db8:1::2"),
				ParseIP("10.1.2.4"),
			},
			want: []IPAddr{
				{IP: ParseIP("2001:db8:1::1")},
				{IP: ParseIP("10.1.2.3")},
			},
			reverse: true,
		},

		// Prefer smaller scope.
		{
			in: []IPAddr{
				{IP: ParseIP("2001:db8:1::1")},
				{IP: ParseIP("fe80::1")},
			},
			srcs: []IP{
				ParseIP("2001:db8:1::2"),
				ParseIP("fe80::2"),
			},
			want: []IPAddr{
				{IP: ParseIP("fe80::1")},
				{IP: ParseIP("2001:db8:1::1")},
			},
			reverse: true,
		},

		// Issue 13283.  Having a 10/8 source address does not
		// mean we should prefer 23/8 destination addresses.
		{
			in: []IPAddr{
				{IP: ParseIP("54.83.193.112")},
				{IP: ParseIP("184.72.238.214")},
				{IP: ParseIP("23.23.172.185")},
				{IP: ParseIP("75.101.148.21")},
				{IP: ParseIP("23.23.134.56")},
				{IP: ParseIP("23.21.50.150")},
			},
			srcs: []IP{
				ParseIP("10.2.3.4"),
				ParseIP("10.2.3.4"),
				ParseIP("10.2.3.4"),
				ParseIP("10.2.3.4"),
				ParseIP("10.2.3.4"),
				ParseIP("10.2.3.4"),
			},
			want: []IPAddr{
				{IP: ParseIP("54.83.193.112")},
				{IP: ParseIP("184.72.238.214")},
				{IP: ParseIP("23.23.172.185")},
				{IP: ParseIP("75.101.148.21")},
				{IP: ParseIP("23.23.134.56")},
				{IP: ParseIP("23.21.50.150")},
			},
			reverse: false,
		},
	}
	for i, tt := range tests {
		inCopy := make([]IPAddr, len(tt.in))
		copy(inCopy, tt.in)
		srcCopy := make([]IP, len(tt.in))
		copy(srcCopy, tt.srcs)
		sortByRFC6724withSrcs(inCopy, srcCopy)
		if !reflect.DeepEqual(inCopy, tt.want) {
			t.Errorf("test %d:\nin = %s\ngot: %s\nwant: %s\n", i, tt.in, inCopy, tt.want)
		}
		if tt.reverse {
			copy(inCopy, tt.in)
			copy(srcCopy, tt.srcs)
			for j := 0; j < len(inCopy)/2; j++ {
				k := len(inCopy) - j - 1
				inCopy[j], inCopy[k] = inCopy[k], inCopy[j]
				srcCopy[j], srcCopy[k] = srcCopy[k], srcCopy[j]
			}
			sortByRFC6724withSrcs(inCopy, srcCopy)
			if !reflect.DeepEqual(inCopy, tt.want) {
				t.Errorf("test %d, starting backwards:\nin = %s\ngot: %s\nwant: %s\n", i, tt.in, inCopy, tt.want)
			}
		}

	}

}

func TestRFC6724PolicyTableClassify(t *testing.T) {
	tests := []struct {
		ip   IP
		want policyTableEntry
	}{
		{
			ip: ParseIP("127.0.0.1"),
			want: policyTableEntry{
				Prefix:     &IPNet{IP: ParseIP("::ffff:0:0"), Mask: CIDRMask(96, 128)},
				Precedence: 35,
				Label:      4,
			},
		},
		{
			ip: ParseIP("2601:645:8002:a500:986f:1db8:c836:bd65"),
			want: policyTableEntry{
				Prefix:     &IPNet{IP: ParseIP("::"), Mask: CIDRMask(0, 128)},
				Precedence: 40,
				Label:      1,
			},
		},
		{
			ip: ParseIP("::1"),
			want: policyTableEntry{
				Prefix:     &IPNet{IP: ParseIP("::1"), Mask: CIDRMask(128, 128)},
				Precedence: 50,
				Label:      0,
			},
		},
		{
			ip: ParseIP("2002::ab12"),
			want: policyTableEntry{
				Prefix:     &IPNet{IP: ParseIP("2002::"), Mask: CIDRMask(16, 128)},
				Precedence: 30,
				Label:      2,
			},
		},
	}
	for i, tt := range tests {
		got := rfc6724policyTable.Classify(tt.ip)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%d. Classify(%s) = %v; want %v", i, tt.ip, got, tt.want)
		}
	}
}

func TestRFC6724ClassifyScope(t *testing.T) {
	tests := []struct {
		ip   IP
		want scope
	}{
		{ParseIP("127.0.0.1"), scopeLinkLocal},   // rfc6724#section-3.2
		{ParseIP("::1"), scopeLinkLocal},         // rfc4007#section-4
		{ParseIP("169.254.1.2"), scopeLinkLocal}, // rfc6724#section-3.2
		{ParseIP("fec0::1"), scopeSiteLocal},
		{ParseIP("8.8.8.8"), scopeGlobal},

		{ParseIP("ff02::"), scopeLinkLocal},  // IPv6 multicast
		{ParseIP("ff05::"), scopeSiteLocal},  // IPv6 multicast
		{ParseIP("ff04::"), scopeAdminLocal}, // IPv6 multicast
		{ParseIP("ff0e::"), scopeGlobal},     // IPv6 multicast

		{IPv4(0xe0, 0, 0, 0), scopeGlobal},       // IPv4 link-local multicast as 16 bytes
		{IPv4(0xe0, 2, 2, 2), scopeGlobal},       // IPv4 global multicast as 16 bytes
		{IPv4(0xe0, 0, 0, 0).To4(), scopeGlobal}, // IPv4 link-local multicast as 4 bytes
		{IPv4(0xe0, 2, 2, 2).To4(), scopeGlobal}, // IPv4 global multicast as 4 bytes
	}
	for i, tt := range tests {
		got := classifyScope(tt.ip)
		if got != tt.want {
			t.Errorf("%d. classifyScope(%s) = %x; want %x", i, tt.ip, got, tt.want)
		}
	}
}

func TestRFC6724CommonPrefixLength(t *testing.T) {
	tests := []struct {
		a, b IP
		want int
	}{
		{ParseIP("fe80::1"), ParseIP("fe80::2"), 64},
		{ParseIP("fe81::1"), ParseIP("fe80::2"), 15},
		{ParseIP("127.0.0.1"), ParseIP("fe80::1"), 0}, // diff size
		{IPv4(1, 2, 3, 4), IP{1, 2, 3, 4}, 32},
		{IP{1, 2, 255, 255}, IP{1, 2, 0, 0}, 16},
		{IP{1, 2, 127, 255}, IP{1, 2, 0, 0}, 17},
		{IP{1, 2, 63, 255}, IP{1, 2, 0, 0}, 18},
		{IP{1, 2, 31, 255}, IP{1, 2, 0, 0}, 19},
		{IP{1, 2, 15, 255}, IP{1, 2, 0, 0}, 20},
		{IP{1, 2, 7, 255}, IP{1, 2, 0, 0}, 21},
		{IP{1, 2, 3, 255}, IP{1, 2, 0, 0}, 22},
		{IP{1, 2, 1, 255}, IP{1, 2, 0, 0}, 23},
		{IP{1, 2, 0, 255}, IP{1, 2, 0, 0}, 24},
	}
	for i, tt := range tests {
		got := commonPrefixLen(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("%d. commonPrefixLen(%s, %s) = %d; want %d", i, tt.a, tt.b, got, tt.want)
		}
	}

}
