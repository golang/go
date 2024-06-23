// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netip_test

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"internal/testenv"
	"net"
	. "net/netip"
	"reflect"
	"slices"
	"strings"
	"testing"
	"unique"
)

var long = flag.Bool("long", false, "run long tests")

type uint128 = Uint128

var (
	mustPrefix = MustParsePrefix
	mustIP     = MustParseAddr
	mustIPPort = MustParseAddrPort
)

func TestParseAddr(t *testing.T) {
	var validIPs = []struct {
		in      string
		ip      Addr   // output of ParseAddr()
		str     string // output of String(). If "", use in.
		wantErr string
	}{
		// Basic zero IPv4 address.
		{
			in: "0.0.0.0",
			ip: MkAddr(Mk128(0, 0xffff00000000), Z4),
		},
		// Basic non-zero IPv4 address.
		{
			in: "192.168.140.255",
			ip: MkAddr(Mk128(0, 0xffffc0a88cff), Z4),
		},
		// IPv4 address in windows-style "print all the digits" form.
		{
			in:      "010.000.015.001",
			wantErr: `ParseAddr("010.000.015.001"): IPv4 field has octet with leading zero`,
		},
		// IPv4 address with a silly amount of leading zeros.
		{
			in:      "000001.00000002.00000003.000000004",
			wantErr: `ParseAddr("000001.00000002.00000003.000000004"): IPv4 field has octet with leading zero`,
		},
		// 4-in-6 with octet with leading zero
		{
			in:      "::ffff:1.2.03.4",
			wantErr: `ParseAddr("::ffff:1.2.03.4"): IPv4 field has octet with leading zero`,
		},
		// 4-in-6 with octet with unexpected character
		{
			in:      "::ffff:1.2.3.z",
			wantErr: `ParseAddr("::ffff:1.2.3.z"): unexpected character (at "z")`,
		},
		// Basic zero IPv6 address.
		{
			in: "::",
			ip: MkAddr(Mk128(0, 0), Z6noz),
		},
		// Localhost IPv6.
		{
			in: "::1",
			ip: MkAddr(Mk128(0, 1), Z6noz),
		},
		// Fully expanded IPv6 address.
		{
			in: "fd7a:115c:a1e0:ab12:4843:cd96:626b:430b",
			ip: MkAddr(Mk128(0xfd7a115ca1e0ab12, 0x4843cd96626b430b), Z6noz),
		},
		// IPv6 with elided fields in the middle.
		{
			in: "fd7a:115c::626b:430b",
			ip: MkAddr(Mk128(0xfd7a115c00000000, 0x00000000626b430b), Z6noz),
		},
		// IPv6 with elided fields at the end.
		{
			in: "fd7a:115c:a1e0:ab12:4843:cd96::",
			ip: MkAddr(Mk128(0xfd7a115ca1e0ab12, 0x4843cd9600000000), Z6noz),
		},
		// IPv6 with single elided field at the end.
		{
			in:  "fd7a:115c:a1e0:ab12:4843:cd96:626b::",
			ip:  MkAddr(Mk128(0xfd7a115ca1e0ab12, 0x4843cd96626b0000), Z6noz),
			str: "fd7a:115c:a1e0:ab12:4843:cd96:626b:0",
		},
		// IPv6 with single elided field in the middle.
		{
			in:  "fd7a:115c:a1e0::4843:cd96:626b:430b",
			ip:  MkAddr(Mk128(0xfd7a115ca1e00000, 0x4843cd96626b430b), Z6noz),
			str: "fd7a:115c:a1e0:0:4843:cd96:626b:430b",
		},
		// IPv6 with the trailing 32 bits written as IPv4 dotted decimal. (4in6)
		{
			in:  "::ffff:192.168.140.255",
			ip:  MkAddr(Mk128(0, 0x0000ffffc0a88cff), Z6noz),
			str: "::ffff:192.168.140.255",
		},
		// IPv6 with a zone specifier.
		{
			in: "fd7a:115c:a1e0:ab12:4843:cd96:626b:430b%eth0",
			ip: MkAddr(Mk128(0xfd7a115ca1e0ab12, 0x4843cd96626b430b), unique.Make(MakeAddrDetail(true, "eth0"))),
		},
		// IPv6 with dotted decimal and zone specifier.
		{
			in:  "1:2::ffff:192.168.140.255%eth1",
			ip:  MkAddr(Mk128(0x0001000200000000, 0x0000ffffc0a88cff), unique.Make(MakeAddrDetail(true, "eth1"))),
			str: "1:2::ffff:c0a8:8cff%eth1",
		},
		// 4-in-6 with zone
		{
			in:  "::ffff:192.168.140.255%eth1",
			ip:  MkAddr(Mk128(0, 0x0000ffffc0a88cff), unique.Make(MakeAddrDetail(true, "eth1"))),
			str: "::ffff:192.168.140.255%eth1",
		},
		// IPv6 with capital letters.
		{
			in:  "FD9E:1A04:F01D::1",
			ip:  MkAddr(Mk128(0xfd9e1a04f01d0000, 0x1), Z6noz),
			str: "fd9e:1a04:f01d::1",
		},
	}

	for _, test := range validIPs {
		t.Run(test.in, func(t *testing.T) {
			got, err := ParseAddr(test.in)
			if err != nil {
				if err.Error() == test.wantErr {
					return
				}
				t.Fatal(err)
			}
			if test.wantErr != "" {
				t.Fatalf("wanted error %q; got none", test.wantErr)
			}
			if got != test.ip {
				t.Errorf("got %#v, want %#v", got, test.ip)
			}

			// Check that ParseAddr is a pure function.
			got2, err := ParseAddr(test.in)
			if err != nil {
				t.Fatal(err)
			}
			if got != got2 {
				t.Errorf("ParseAddr(%q) got 2 different results: %#v, %#v", test.in, got, got2)
			}

			// Check that ParseAddr(ip.String()) is the identity function.
			s := got.String()
			got3, err := ParseAddr(s)
			if err != nil {
				t.Fatal(err)
			}
			if got != got3 {
				t.Errorf("ParseAddr(%q) != ParseAddr(ParseIP(%q).String()). Got %#v, want %#v", test.in, test.in, got3, got)
			}

			// Check that the slow-but-readable parser produces the same result.
			slow, err := parseIPSlow(test.in)
			if err != nil {
				t.Fatal(err)
			}
			if got != slow {
				t.Errorf("ParseAddr(%q) = %#v, parseIPSlow(%q) = %#v", test.in, got, test.in, slow)
			}

			// Check that the parsed IP formats as expected.
			s = got.String()
			wants := test.str
			if wants == "" {
				wants = test.in
			}
			if s != wants {
				t.Errorf("ParseAddr(%q).String() got %q, want %q", test.in, s, wants)
			}

			// Check that AppendTo matches MarshalText.
			TestAppendToMarshal(t, got)

			// Check that MarshalText/UnmarshalText work similarly to
			// ParseAddr/String (see TestIPMarshalUnmarshal for
			// marshal-specific behavior that's not common with
			// ParseAddr/String).
			js := `"` + test.in + `"`
			var jsgot Addr
			if err := json.Unmarshal([]byte(js), &jsgot); err != nil {
				t.Fatal(err)
			}
			if jsgot != got {
				t.Errorf("json.Unmarshal(%q) = %#v, want %#v", test.in, jsgot, got)
			}
			jsb, err := json.Marshal(jsgot)
			if err != nil {
				t.Fatal(err)
			}
			jswant := `"` + wants + `"`
			jsback := string(jsb)
			if jsback != jswant {
				t.Errorf("Marshal(Unmarshal(%q)) = %s, want %s", test.in, jsback, jswant)
			}
		})
	}

	var invalidIPs = []string{
		// Empty string
		"",
		// Garbage non-IP
		"bad",
		// Single number. Some parsers accept this as an IPv4 address in
		// big-endian uint32 form, but we don't.
		"1234",
		// IPv4 with a zone specifier
		"1.2.3.4%eth0",
		// IPv4 field must have at least one digit
		".1.2.3",
		"1.2.3.",
		"1..2.3",
		// IPv4 address too long
		"1.2.3.4.5",
		// IPv4 in dotted octal form
		"0300.0250.0214.0377",
		// IPv4 in dotted hex form
		"0xc0.0xa8.0x8c.0xff",
		// IPv4 in class B form
		"192.168.12345",
		// IPv4 in class B form, with a small enough number to be
		// parseable as a regular dotted decimal field.
		"127.0.1",
		// IPv4 in class A form
		"192.1234567",
		// IPv4 in class A form, with a small enough number to be
		// parseable as a regular dotted decimal field.
		"127.1",
		// IPv4 field has value >255
		"192.168.300.1",
		// IPv4 with too many fields
		"192.168.0.1.5.6",
		// IPv6 with not enough fields
		"1:2:3:4:5:6:7",
		// IPv6 with too many fields
		"1:2:3:4:5:6:7:8:9",
		// IPv6 with 8 fields and a :: expander
		"1:2:3:4::5:6:7:8",
		// IPv6 with a field bigger than 2b
		"fe801::1",
		// IPv6 with non-hex values in field
		"fe80:tail:scal:e::",
		// IPv6 with a zone delimiter but no zone.
		"fe80::1%",
		// IPv6 (without ellipsis) with too many fields for trailing embedded IPv4.
		"ffff:ffff:ffff:ffff:ffff:ffff:ffff:192.168.140.255",
		// IPv6 (with ellipsis) with too many fields for trailing embedded IPv4.
		"ffff::ffff:ffff:ffff:ffff:ffff:ffff:192.168.140.255",
		// IPv6 with invalid embedded IPv4.
		"::ffff:192.168.140.bad",
		// IPv6 with multiple ellipsis ::.
		"fe80::1::1",
		// IPv6 with invalid non hex/colon character.
		"fe80:1?:1",
		// IPv6 with truncated bytes after single colon.
		"fe80:",
		// IPv6 with 5 zeros in last group
		"0:0:0:0:0:ffff:0:00000",
		// IPv6 with 5 zeros in one group and embedded IPv4
		"0:0:0:0:00000:ffff:127.1.2.3",
	}

	for _, s := range invalidIPs {
		t.Run(s, func(t *testing.T) {
			got, err := ParseAddr(s)
			if err == nil {
				t.Errorf("ParseAddr(%q) = %#v, want error", s, got)
			}

			slow, err := parseIPSlow(s)
			if err == nil {
				t.Errorf("parseIPSlow(%q) = %#v, want error", s, slow)
			}

			std := net.ParseIP(s)
			if std != nil {
				t.Errorf("net.ParseIP(%q) = %#v, want error", s, std)
			}

			if s == "" {
				// Don't test unmarshaling of "" here, do it in
				// IPMarshalUnmarshal.
				return
			}
			var jsgot Addr
			js := []byte(`"` + s + `"`)
			if err := json.Unmarshal(js, &jsgot); err == nil {
				t.Errorf("json.Unmarshal(%q) = %#v, want error", s, jsgot)
			}
		})
	}
}

func TestAddrFromSlice(t *testing.T) {
	tests := []struct {
		ip       []byte
		wantAddr Addr
		wantOK   bool
	}{
		{
			ip:       []byte{10, 0, 0, 1},
			wantAddr: mustIP("10.0.0.1"),
			wantOK:   true,
		},
		{
			ip:       []byte{0xfe, 0x80, 15: 0x01},
			wantAddr: mustIP("fe80::01"),
			wantOK:   true,
		},
		{
			ip:       []byte{0, 1, 2},
			wantAddr: Addr{},
			wantOK:   false,
		},
		{
			ip:       nil,
			wantAddr: Addr{},
			wantOK:   false,
		},
	}
	for _, tt := range tests {
		addr, ok := AddrFromSlice(tt.ip)
		if ok != tt.wantOK || addr != tt.wantAddr {
			t.Errorf("AddrFromSlice(%#v) = %#v, %v, want %#v, %v", tt.ip, addr, ok, tt.wantAddr, tt.wantOK)
		}
	}
}

func TestIPv4Constructors(t *testing.T) {
	if AddrFrom4([4]byte{1, 2, 3, 4}) != MustParseAddr("1.2.3.4") {
		t.Errorf("don't match")
	}
}

func TestAddrMarshalUnmarshalBinary(t *testing.T) {
	tests := []struct {
		ip       string
		wantSize int
	}{
		{"", 0}, // zero IP
		{"1.2.3.4", 4},
		{"fd7a:115c:a1e0:ab12:4843:cd96:626b:430b", 16},
		{"::ffff:c000:0280", 16},
		{"::ffff:c000:0280%eth0", 20},
	}
	for _, tc := range tests {
		var ip Addr
		if len(tc.ip) > 0 {
			ip = mustIP(tc.ip)
		}
		b, err := ip.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		if len(b) != tc.wantSize {
			t.Fatalf("%q encoded to size %d; want %d", tc.ip, len(b), tc.wantSize)
		}
		var ip2 Addr
		if err := ip2.UnmarshalBinary(b); err != nil {
			t.Fatal(err)
		}
		if ip != ip2 {
			t.Fatalf("got %v; want %v", ip2, ip)
		}
	}

	// Cannot unmarshal from unexpected IP length.
	for _, n := range []int{3, 5} {
		var ip2 Addr
		if err := ip2.UnmarshalBinary(bytes.Repeat([]byte{1}, n)); err == nil {
			t.Fatalf("unmarshaled from unexpected IP length %d", n)
		}
	}
}

func TestAddrPortMarshalTextString(t *testing.T) {
	tests := []struct {
		in   AddrPort
		want string
	}{
		{mustIPPort("1.2.3.4:80"), "1.2.3.4:80"},
		{mustIPPort("[::]:80"), "[::]:80"},
		{mustIPPort("[1::CAFE]:80"), "[1::cafe]:80"},
		{mustIPPort("[1::CAFE%en0]:80"), "[1::cafe%en0]:80"},
		{mustIPPort("[::FFFF:192.168.140.255]:80"), "[::ffff:192.168.140.255]:80"},
		{mustIPPort("[::FFFF:192.168.140.255%en0]:80"), "[::ffff:192.168.140.255%en0]:80"},
	}
	for i, tt := range tests {
		if got := tt.in.String(); got != tt.want {
			t.Errorf("%d. for (%v, %v) String = %q; want %q", i, tt.in.Addr(), tt.in.Port(), got, tt.want)
		}
		mt, err := tt.in.MarshalText()
		if err != nil {
			t.Errorf("%d. for (%v, %v) MarshalText error: %v", i, tt.in.Addr(), tt.in.Port(), err)
			continue
		}
		if string(mt) != tt.want {
			t.Errorf("%d. for (%v, %v) MarshalText = %q; want %q", i, tt.in.Addr(), tt.in.Port(), mt, tt.want)
		}
	}
}

func TestAddrPortMarshalUnmarshalBinary(t *testing.T) {
	tests := []struct {
		ipport   string
		wantSize int
	}{
		{"1.2.3.4:51820", 4 + 2},
		{"[fd7a:115c:a1e0:ab12:4843:cd96:626b:430b]:80", 16 + 2},
		{"[::ffff:c000:0280]:65535", 16 + 2},
		{"[::ffff:c000:0280%eth0]:1", 20 + 2},
	}
	for _, tc := range tests {
		var ipport AddrPort
		if len(tc.ipport) > 0 {
			ipport = mustIPPort(tc.ipport)
		}
		b, err := ipport.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		if len(b) != tc.wantSize {
			t.Fatalf("%q encoded to size %d; want %d", tc.ipport, len(b), tc.wantSize)
		}
		var ipport2 AddrPort
		if err := ipport2.UnmarshalBinary(b); err != nil {
			t.Fatal(err)
		}
		if ipport != ipport2 {
			t.Fatalf("got %v; want %v", ipport2, ipport)
		}
	}

	// Cannot unmarshal from unexpected lengths.
	for _, n := range []int{3, 7} {
		var ipport2 AddrPort
		if err := ipport2.UnmarshalBinary(bytes.Repeat([]byte{1}, n)); err == nil {
			t.Fatalf("unmarshaled from unexpected length %d", n)
		}
	}
}

func TestPrefixMarshalTextString(t *testing.T) {
	tests := []struct {
		in   Prefix
		want string
	}{
		{mustPrefix("1.2.3.4/24"), "1.2.3.4/24"},
		{mustPrefix("fd7a:115c:a1e0:ab12:4843:cd96:626b:430b/118"), "fd7a:115c:a1e0:ab12:4843:cd96:626b:430b/118"},
		{mustPrefix("::ffff:c000:0280/96"), "::ffff:192.0.2.128/96"},
		{mustPrefix("::ffff:192.168.140.255/8"), "::ffff:192.168.140.255/8"},
		{PrefixFrom(mustIP("::ffff:c000:0280").WithZone("eth0"), 37), "::ffff:192.0.2.128/37"}, // Zone should be stripped
	}
	for i, tt := range tests {
		if got := tt.in.String(); got != tt.want {
			t.Errorf("%d. for %v String = %q; want %q", i, tt.in, got, tt.want)
		}
		mt, err := tt.in.MarshalText()
		if err != nil {
			t.Errorf("%d. for %v MarshalText error: %v", i, tt.in, err)
			continue
		}
		if string(mt) != tt.want {
			t.Errorf("%d. for %v MarshalText = %q; want %q", i, tt.in, mt, tt.want)
		}
	}
}

func TestPrefixMarshalUnmarshalBinary(t *testing.T) {
	type testCase struct {
		prefix   Prefix
		wantSize int
	}
	tests := []testCase{
		{mustPrefix("1.2.3.4/24"), 4 + 1},
		{mustPrefix("fd7a:115c:a1e0:ab12:4843:cd96:626b:430b/118"), 16 + 1},
		{mustPrefix("::ffff:c000:0280/96"), 16 + 1},
		{PrefixFrom(mustIP("::ffff:c000:0280").WithZone("eth0"), 37), 16 + 1}, // Zone should be stripped
	}
	tests = append(tests,
		testCase{PrefixFrom(tests[0].prefix.Addr(), 33), tests[0].wantSize},
		testCase{PrefixFrom(tests[1].prefix.Addr(), 129), tests[1].wantSize})
	for _, tc := range tests {
		prefix := tc.prefix
		b, err := prefix.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		if len(b) != tc.wantSize {
			t.Fatalf("%q encoded to size %d; want %d", tc.prefix, len(b), tc.wantSize)
		}
		var prefix2 Prefix
		if err := prefix2.UnmarshalBinary(b); err != nil {
			t.Fatal(err)
		}
		if prefix != prefix2 {
			t.Fatalf("got %v; want %v", prefix2, prefix)
		}
	}

	// Cannot unmarshal from unexpected lengths.
	for _, n := range []int{3, 6} {
		var prefix2 Prefix
		if err := prefix2.UnmarshalBinary(bytes.Repeat([]byte{1}, n)); err == nil {
			t.Fatalf("unmarshaled from unexpected length %d", n)
		}
	}
}

func TestAddrMarshalUnmarshal(t *testing.T) {
	// This only tests the cases where Marshal/Unmarshal diverges from
	// the behavior of ParseAddr/String. For the rest of the test cases,
	// see TestParseAddr above.
	orig := `""`
	var ip Addr
	if err := json.Unmarshal([]byte(orig), &ip); err != nil {
		t.Fatalf("Unmarshal(%q) got error %v", orig, err)
	}
	if ip != (Addr{}) {
		t.Errorf("Unmarshal(%q) is not the zero Addr", orig)
	}

	jsb, err := json.Marshal(ip)
	if err != nil {
		t.Fatalf("Marshal(%v) got error %v", ip, err)
	}
	back := string(jsb)
	if back != orig {
		t.Errorf("Marshal(Unmarshal(%q)) got %q, want %q", orig, back, orig)
	}
}

func TestAddrFrom16(t *testing.T) {
	tests := []struct {
		name string
		in   [16]byte
		want Addr
	}{
		{
			name: "v6-raw",
			in:   [...]byte{15: 1},
			want: MkAddr(Mk128(0, 1), Z6noz),
		},
		{
			name: "v4-raw",
			in:   [...]byte{10: 0xff, 11: 0xff, 12: 1, 13: 2, 14: 3, 15: 4},
			want: MkAddr(Mk128(0, 0xffff01020304), Z6noz),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AddrFrom16(tt.in)
			if got != tt.want {
				t.Errorf("got %#v; want %#v", got, tt.want)
			}
		})
	}
}

func TestIPProperties(t *testing.T) {
	var (
		nilIP Addr

		unicast4           = mustIP("192.0.2.1")
		unicast6           = mustIP("2001:db8::1")
		unicastZone6       = mustIP("2001:db8::1%eth0")
		unicast6Unassigned = mustIP("4000::1") // not in 2000::/3.

		multicast4     = mustIP("224.0.0.1")
		multicast6     = mustIP("ff02::1")
		multicastZone6 = mustIP("ff02::1%eth0")

		llu4     = mustIP("169.254.0.1")
		llu6     = mustIP("fe80::1")
		llu6Last = mustIP("febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff")
		lluZone6 = mustIP("fe80::1%eth0")

		loopback4 = mustIP("127.0.0.1")

		ilm6     = mustIP("ff01::1")
		ilmZone6 = mustIP("ff01::1%eth0")

		private4a        = mustIP("10.0.0.1")
		private4b        = mustIP("172.16.0.1")
		private4c        = mustIP("192.168.1.1")
		private6         = mustIP("fd00::1")
		private6mapped4a = mustIP("::ffff:10.0.0.1")
		private6mapped4b = mustIP("::ffff:172.16.0.1")
		private6mapped4c = mustIP("::ffff:192.168.1.1")
	)

	tests := []struct {
		name                    string
		ip                      Addr
		globalUnicast           bool
		interfaceLocalMulticast bool
		linkLocalMulticast      bool
		linkLocalUnicast        bool
		loopback                bool
		multicast               bool
		private                 bool
		unspecified             bool
	}{
		{
			name: "nil",
			ip:   nilIP,
		},
		{
			name:          "unicast v4Addr",
			ip:            unicast4,
			globalUnicast: true,
		},
		{
			name:          "unicast v6 mapped v4Addr",
			ip:            AddrFrom16(unicast4.As16()),
			globalUnicast: true,
		},
		{
			name:          "unicast v6Addr",
			ip:            unicast6,
			globalUnicast: true,
		},
		{
			name:          "unicast v6AddrZone",
			ip:            unicastZone6,
			globalUnicast: true,
		},
		{
			name:          "unicast v6Addr unassigned",
			ip:            unicast6Unassigned,
			globalUnicast: true,
		},
		{
			name:               "multicast v4Addr",
			ip:                 multicast4,
			linkLocalMulticast: true,
			multicast:          true,
		},
		{
			name:               "multicast v6 mapped v4Addr",
			ip:                 AddrFrom16(multicast4.As16()),
			linkLocalMulticast: true,
			multicast:          true,
		},
		{
			name:               "multicast v6Addr",
			ip:                 multicast6,
			linkLocalMulticast: true,
			multicast:          true,
		},
		{
			name:               "multicast v6AddrZone",
			ip:                 multicastZone6,
			linkLocalMulticast: true,
			multicast:          true,
		},
		{
			name:             "link-local unicast v4Addr",
			ip:               llu4,
			linkLocalUnicast: true,
		},
		{
			name:             "link-local unicast v6 mapped v4Addr",
			ip:               AddrFrom16(llu4.As16()),
			linkLocalUnicast: true,
		},
		{
			name:             "link-local unicast v6Addr",
			ip:               llu6,
			linkLocalUnicast: true,
		},
		{
			name:             "link-local unicast v6Addr upper bound",
			ip:               llu6Last,
			linkLocalUnicast: true,
		},
		{
			name:             "link-local unicast v6AddrZone",
			ip:               lluZone6,
			linkLocalUnicast: true,
		},
		{
			name:     "loopback v4Addr",
			ip:       loopback4,
			loopback: true,
		},
		{
			name:     "loopback v6Addr",
			ip:       IPv6Loopback(),
			loopback: true,
		},
		{
			name:     "loopback v6 mapped v4Addr",
			ip:       AddrFrom16(IPv6Loopback().As16()),
			loopback: true,
		},
		{
			name:                    "interface-local multicast v6Addr",
			ip:                      ilm6,
			interfaceLocalMulticast: true,
			multicast:               true,
		},
		{
			name:                    "interface-local multicast v6AddrZone",
			ip:                      ilmZone6,
			interfaceLocalMulticast: true,
			multicast:               true,
		},
		{
			name:          "private v4Addr 10/8",
			ip:            private4a,
			globalUnicast: true,
			private:       true,
		},
		{
			name:          "private v4Addr 172.16/12",
			ip:            private4b,
			globalUnicast: true,
			private:       true,
		},
		{
			name:          "private v4Addr 192.168/16",
			ip:            private4c,
			globalUnicast: true,
			private:       true,
		},
		{
			name:          "private v6Addr",
			ip:            private6,
			globalUnicast: true,
			private:       true,
		},
		{
			name:          "private v6 mapped v4Addr 10/8",
			ip:            private6mapped4a,
			globalUnicast: true,
			private:       true,
		},
		{
			name:          "private v6 mapped v4Addr 172.16/12",
			ip:            private6mapped4b,
			globalUnicast: true,
			private:       true,
		},
		{
			name:          "private v6 mapped v4Addr 192.168/16",
			ip:            private6mapped4c,
			globalUnicast: true,
			private:       true,
		},
		{
			name:        "unspecified v4Addr",
			ip:          IPv4Unspecified(),
			unspecified: true,
		},
		{
			name:        "unspecified v6Addr",
			ip:          IPv6Unspecified(),
			unspecified: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gu := tt.ip.IsGlobalUnicast()
			if gu != tt.globalUnicast {
				t.Errorf("IsGlobalUnicast(%v) = %v; want %v", tt.ip, gu, tt.globalUnicast)
			}

			ilm := tt.ip.IsInterfaceLocalMulticast()
			if ilm != tt.interfaceLocalMulticast {
				t.Errorf("IsInterfaceLocalMulticast(%v) = %v; want %v", tt.ip, ilm, tt.interfaceLocalMulticast)
			}

			llu := tt.ip.IsLinkLocalUnicast()
			if llu != tt.linkLocalUnicast {
				t.Errorf("IsLinkLocalUnicast(%v) = %v; want %v", tt.ip, llu, tt.linkLocalUnicast)
			}

			llm := tt.ip.IsLinkLocalMulticast()
			if llm != tt.linkLocalMulticast {
				t.Errorf("IsLinkLocalMulticast(%v) = %v; want %v", tt.ip, llm, tt.linkLocalMulticast)
			}

			lo := tt.ip.IsLoopback()
			if lo != tt.loopback {
				t.Errorf("IsLoopback(%v) = %v; want %v", tt.ip, lo, tt.loopback)
			}

			multicast := tt.ip.IsMulticast()
			if multicast != tt.multicast {
				t.Errorf("IsMulticast(%v) = %v; want %v", tt.ip, multicast, tt.multicast)
			}

			private := tt.ip.IsPrivate()
			if private != tt.private {
				t.Errorf("IsPrivate(%v) = %v; want %v", tt.ip, private, tt.private)
			}

			unspecified := tt.ip.IsUnspecified()
			if unspecified != tt.unspecified {
				t.Errorf("IsUnspecified(%v) = %v; want %v", tt.ip, unspecified, tt.unspecified)
			}
		})
	}
}

func TestAddrWellKnown(t *testing.T) {
	tests := []struct {
		name string
		ip   Addr
		std  net.IP
	}{
		{
			name: "IPv4 unspecified",
			ip:   IPv4Unspecified(),
			std:  net.IPv4zero,
		},
		{
			name: "IPv6 link-local all nodes",
			ip:   IPv6LinkLocalAllNodes(),
			std:  net.IPv6linklocalallnodes,
		},
		{
			name: "IPv6 link-local all routers",
			ip:   IPv6LinkLocalAllRouters(),
			std:  net.IPv6linklocalallrouters,
		},
		{
			name: "IPv6 loopback",
			ip:   IPv6Loopback(),
			std:  net.IPv6loopback,
		},
		{
			name: "IPv6 unspecified",
			ip:   IPv6Unspecified(),
			std:  net.IPv6unspecified,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			want := tt.std.String()
			got := tt.ip.String()

			if got != want {
				t.Fatalf("got %s, want %s", got, want)
			}
		})
	}
}

func TestAddrLessCompare(t *testing.T) {
	tests := []struct {
		a, b Addr
		want bool
	}{
		{Addr{}, Addr{}, false},
		{Addr{}, mustIP("1.2.3.4"), true},
		{mustIP("1.2.3.4"), Addr{}, false},

		{mustIP("1.2.3.4"), mustIP("0102:0304::0"), true},
		{mustIP("0102:0304::0"), mustIP("1.2.3.4"), false},
		{mustIP("1.2.3.4"), mustIP("1.2.3.4"), false},

		{mustIP("::1"), mustIP("::2"), true},
		{mustIP("::1"), mustIP("::1%foo"), true},
		{mustIP("::1%foo"), mustIP("::2"), true},
		{mustIP("::2"), mustIP("::3"), true},

		{mustIP("::"), mustIP("0.0.0.0"), false},
		{mustIP("0.0.0.0"), mustIP("::"), true},

		{mustIP("::1%a"), mustIP("::1%b"), true},
		{mustIP("::1%a"), mustIP("::1%a"), false},
		{mustIP("::1%b"), mustIP("::1%a"), false},
	}
	for _, tt := range tests {
		got := tt.a.Less(tt.b)
		if got != tt.want {
			t.Errorf("Less(%q, %q) = %v; want %v", tt.a, tt.b, got, tt.want)
		}
		cmp := tt.a.Compare(tt.b)
		if got && cmp != -1 {
			t.Errorf("Less(%q, %q) = true, but Compare = %v (not -1)", tt.a, tt.b, cmp)
		}
		if cmp < -1 || cmp > 1 {
			t.Errorf("bogus Compare return value %v", cmp)
		}
		if cmp == 0 && tt.a != tt.b {
			t.Errorf("Compare(%q, %q) = 0; but not equal", tt.a, tt.b)
		}
		if cmp == 1 && !tt.b.Less(tt.a) {
			t.Errorf("Compare(%q, %q) = 1; but b.Less(a) isn't true", tt.a, tt.b)
		}

		// Also check inverse.
		if got == tt.want && got {
			got2 := tt.b.Less(tt.a)
			if got2 {
				t.Errorf("Less(%q, %q) was correctly %v, but so was Less(%q, %q)", tt.a, tt.b, got, tt.b, tt.a)
			}
		}
	}

	// And just sort.
	values := []Addr{
		mustIP("::1"),
		mustIP("::2"),
		Addr{},
		mustIP("1.2.3.4"),
		mustIP("8.8.8.8"),
		mustIP("::1%foo"),
	}
	slices.SortFunc(values, Addr.Compare)
	got := fmt.Sprintf("%s", values)
	want := `[invalid IP 1.2.3.4 8.8.8.8 ::1 ::1%foo ::2]`
	if got != want {
		t.Errorf("unexpected sort\n got: %s\nwant: %s\n", got, want)
	}
}

func TestAddrPortCompare(t *testing.T) {
	tests := []struct {
		a, b AddrPort
		want int
	}{
		{AddrPort{}, AddrPort{}, 0},
		{AddrPort{}, mustIPPort("1.2.3.4:80"), -1},

		{mustIPPort("1.2.3.4:80"), mustIPPort("1.2.3.4:80"), 0},
		{mustIPPort("[::1]:80"), mustIPPort("[::1]:80"), 0},

		{mustIPPort("1.2.3.4:80"), mustIPPort("2.3.4.5:22"), -1},
		{mustIPPort("[::1]:80"), mustIPPort("[::2]:22"), -1},

		{mustIPPort("1.2.3.4:80"), mustIPPort("1.2.3.4:443"), -1},
		{mustIPPort("[::1]:80"), mustIPPort("[::1]:443"), -1},

		{mustIPPort("1.2.3.4:80"), mustIPPort("[0102:0304::0]:80"), -1},
	}
	for _, tt := range tests {
		got := tt.a.Compare(tt.b)
		if got != tt.want {
			t.Errorf("Compare(%q, %q) = %v; want %v", tt.a, tt.b, got, tt.want)
		}

		// Also check inverse.
		if got == tt.want {
			got2 := tt.b.Compare(tt.a)
			if want2 := -1 * tt.want; got2 != want2 {
				t.Errorf("Compare(%q, %q) was correctly %v, but Compare(%q, %q) was %v", tt.a, tt.b, got, tt.b, tt.a, got2)
			}
		}
	}

	// And just sort.
	values := []AddrPort{
		mustIPPort("[::1]:80"),
		mustIPPort("[::2]:80"),
		AddrPort{},
		mustIPPort("1.2.3.4:443"),
		mustIPPort("8.8.8.8:8080"),
		mustIPPort("[::1%foo]:1024"),
	}
	slices.SortFunc(values, AddrPort.Compare)
	got := fmt.Sprintf("%s", values)
	want := `[invalid AddrPort 1.2.3.4:443 8.8.8.8:8080 [::1]:80 [::1%foo]:1024 [::2]:80]`
	if got != want {
		t.Errorf("unexpected sort\n got: %s\nwant: %s\n", got, want)
	}
}

func TestPrefixCompare(t *testing.T) {
	tests := []struct {
		a, b Prefix
		want int
	}{
		{Prefix{}, Prefix{}, 0},
		{Prefix{}, mustPrefix("1.2.3.0/24"), -1},

		{mustPrefix("1.2.3.0/24"), mustPrefix("1.2.3.0/24"), 0},
		{mustPrefix("fe80::/64"), mustPrefix("fe80::/64"), 0},

		{mustPrefix("1.2.3.0/24"), mustPrefix("1.2.4.0/24"), -1},
		{mustPrefix("fe80::/64"), mustPrefix("fe90::/64"), -1},

		{mustPrefix("1.2.0.0/16"), mustPrefix("1.2.0.0/24"), -1},
		{mustPrefix("fe80::/48"), mustPrefix("fe80::/64"), -1},

		{mustPrefix("1.2.3.0/24"), mustPrefix("fe80::/8"), -1},
	}
	for _, tt := range tests {
		got := tt.a.Compare(tt.b)
		if got != tt.want {
			t.Errorf("Compare(%q, %q) = %v; want %v", tt.a, tt.b, got, tt.want)
		}

		// Also check inverse.
		if got == tt.want {
			got2 := tt.b.Compare(tt.a)
			if want2 := -1 * tt.want; got2 != want2 {
				t.Errorf("Compare(%q, %q) was correctly %v, but Compare(%q, %q) was %v", tt.a, tt.b, got, tt.b, tt.a, got2)
			}
		}
	}

	// And just sort.
	values := []Prefix{
		mustPrefix("1.2.3.0/24"),
		mustPrefix("fe90::/64"),
		mustPrefix("fe80::/64"),
		mustPrefix("1.2.0.0/16"),
		Prefix{},
		mustPrefix("fe80::/48"),
		mustPrefix("1.2.0.0/24"),
	}
	slices.SortFunc(values, Prefix.Compare)
	got := fmt.Sprintf("%s", values)
	want := `[invalid Prefix 1.2.0.0/16 1.2.0.0/24 1.2.3.0/24 fe80::/48 fe80::/64 fe90::/64]`
	if got != want {
		t.Errorf("unexpected sort\n got: %s\nwant: %s\n", got, want)
	}
}

func TestIPStringExpanded(t *testing.T) {
	tests := []struct {
		ip Addr
		s  string
	}{
		{
			ip: Addr{},
			s:  "invalid IP",
		},
		{
			ip: mustIP("192.0.2.1"),
			s:  "192.0.2.1",
		},
		{
			ip: mustIP("::ffff:192.0.2.1"),
			s:  "0000:0000:0000:0000:0000:ffff:c000:0201",
		},
		{
			ip: mustIP("2001:db8::1"),
			s:  "2001:0db8:0000:0000:0000:0000:0000:0001",
		},
		{
			ip: mustIP("2001:db8::1%eth0"),
			s:  "2001:0db8:0000:0000:0000:0000:0000:0001%eth0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.ip.String(), func(t *testing.T) {
			want := tt.s
			got := tt.ip.StringExpanded()

			if got != want {
				t.Fatalf("got %s, want %s", got, want)
			}
		})
	}
}

func TestPrefixMasking(t *testing.T) {
	type subtest struct {
		ip   Addr
		bits uint8
		p    Prefix
		ok   bool
	}

	// makeIPv6 produces a set of IPv6 subtests with an optional zone identifier.
	makeIPv6 := func(zone string) []subtest {
		if zone != "" {
			zone = "%" + zone
		}

		return []subtest{
			{
				ip:   mustIP(fmt.Sprintf("2001:db8::1%s", zone)),
				bits: 255,
			},
			{
				ip:   mustIP(fmt.Sprintf("2001:db8::1%s", zone)),
				bits: 32,
				p:    mustPrefix("2001:db8::/32"),
				ok:   true,
			},
			{
				ip:   mustIP(fmt.Sprintf("fe80::dead:beef:dead:beef%s", zone)),
				bits: 96,
				p:    mustPrefix("fe80::dead:beef:0:0/96"),
				ok:   true,
			},
			{
				ip:   mustIP(fmt.Sprintf("aaaa::%s", zone)),
				bits: 4,
				p:    mustPrefix("a000::/4"),
				ok:   true,
			},
			{
				ip:   mustIP(fmt.Sprintf("::%s", zone)),
				bits: 63,
				p:    mustPrefix("::/63"),
				ok:   true,
			},
		}
	}

	tests := []struct {
		family   string
		subtests []subtest
	}{
		{
			family: "nil",
			subtests: []subtest{
				{
					bits: 255,
					ok:   true,
				},
				{
					bits: 16,
					ok:   true,
				},
			},
		},
		{
			family: "IPv4",
			subtests: []subtest{
				{
					ip:   mustIP("192.0.2.0"),
					bits: 255,
				},
				{
					ip:   mustIP("192.0.2.0"),
					bits: 16,
					p:    mustPrefix("192.0.0.0/16"),
					ok:   true,
				},
				{
					ip:   mustIP("255.255.255.255"),
					bits: 20,
					p:    mustPrefix("255.255.240.0/20"),
					ok:   true,
				},
				{
					// Partially masking one byte that contains both
					// 1s and 0s on either side of the mask limit.
					ip:   mustIP("100.98.156.66"),
					bits: 10,
					p:    mustPrefix("100.64.0.0/10"),
					ok:   true,
				},
			},
		},
		{
			family:   "IPv6",
			subtests: makeIPv6(""),
		},
		{
			family:   "IPv6 zone",
			subtests: makeIPv6("eth0"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.family, func(t *testing.T) {
			for _, st := range tt.subtests {
				t.Run(st.p.String(), func(t *testing.T) {
					// Ensure st.ip is not mutated.
					orig := st.ip.String()

					p, err := st.ip.Prefix(int(st.bits))
					if st.ok && err != nil {
						t.Fatalf("failed to produce prefix: %v", err)
					}
					if !st.ok && err == nil {
						t.Fatal("expected an error, but none occurred")
					}
					if err != nil {
						t.Logf("err: %v", err)
						return
					}

					if !reflect.DeepEqual(p, st.p) {
						t.Errorf("prefix = %q, want %q", p, st.p)
					}

					if got := st.ip.String(); got != orig {
						t.Errorf("IP was mutated: %q, want %q", got, orig)
					}
				})
			}
		})
	}
}

func TestPrefixMarshalUnmarshal(t *testing.T) {
	tests := []string{
		"",
		"1.2.3.4/32",
		"0.0.0.0/0",
		"::/0",
		"::1/128",
		"2001:db8::/32",
	}

	for _, s := range tests {
		t.Run(s, func(t *testing.T) {
			// Ensure that JSON  (and by extension, text) marshaling is
			// sane by entering quoted input.
			orig := `"` + s + `"`

			var p Prefix
			if err := json.Unmarshal([]byte(orig), &p); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}

			pb, err := json.Marshal(p)
			if err != nil {
				t.Fatalf("failed to marshal: %v", err)
			}

			back := string(pb)
			if orig != back {
				t.Errorf("Marshal = %q; want %q", back, orig)
			}
		})
	}
}

func TestPrefixUnmarshalTextNonZero(t *testing.T) {
	ip := mustPrefix("fe80::/64")
	if err := ip.UnmarshalText([]byte("xxx")); err == nil {
		t.Fatal("unmarshaled into non-empty Prefix")
	}
}

func TestIs4AndIs6(t *testing.T) {
	tests := []struct {
		ip  Addr
		is4 bool
		is6 bool
	}{
		{Addr{}, false, false},
		{mustIP("1.2.3.4"), true, false},
		{mustIP("127.0.0.2"), true, false},
		{mustIP("::1"), false, true},
		{mustIP("::ffff:192.0.2.128"), false, true},
		{mustIP("::fffe:c000:0280"), false, true},
		{mustIP("::1%eth0"), false, true},
	}
	for _, tt := range tests {
		got4 := tt.ip.Is4()
		if got4 != tt.is4 {
			t.Errorf("Is4(%q) = %v; want %v", tt.ip, got4, tt.is4)
		}

		got6 := tt.ip.Is6()
		if got6 != tt.is6 {
			t.Errorf("Is6(%q) = %v; want %v", tt.ip, got6, tt.is6)
		}
	}
}

func TestIs4In6(t *testing.T) {
	tests := []struct {
		ip        Addr
		want      bool
		wantUnmap Addr
	}{
		{Addr{}, false, Addr{}},
		{mustIP("::ffff:c000:0280"), true, mustIP("192.0.2.128")},
		{mustIP("::ffff:192.0.2.128"), true, mustIP("192.0.2.128")},
		{mustIP("::ffff:192.0.2.128%eth0"), true, mustIP("192.0.2.128")},
		{mustIP("::fffe:c000:0280"), false, mustIP("::fffe:c000:0280")},
		{mustIP("::ffff:127.1.2.3"), true, mustIP("127.1.2.3")},
		{mustIP("::ffff:7f01:0203"), true, mustIP("127.1.2.3")},
		{mustIP("0:0:0:0:0000:ffff:127.1.2.3"), true, mustIP("127.1.2.3")},
		{mustIP("0:0:0:0::ffff:127.1.2.3"), true, mustIP("127.1.2.3")},
		{mustIP("::1"), false, mustIP("::1")},
		{mustIP("1.2.3.4"), false, mustIP("1.2.3.4")},
	}
	for _, tt := range tests {
		got := tt.ip.Is4In6()
		if got != tt.want {
			t.Errorf("Is4In6(%q) = %v; want %v", tt.ip, got, tt.want)
		}
		u := tt.ip.Unmap()
		if u != tt.wantUnmap {
			t.Errorf("Unmap(%q) = %v; want %v", tt.ip, u, tt.wantUnmap)
		}
	}
}

func TestPrefixMasked(t *testing.T) {
	tests := []struct {
		prefix Prefix
		masked Prefix
	}{
		{
			prefix: mustPrefix("192.168.0.255/24"),
			masked: mustPrefix("192.168.0.0/24"),
		},
		{
			prefix: mustPrefix("2100::/3"),
			masked: mustPrefix("2000::/3"),
		},
		{
			prefix: PrefixFrom(mustIP("2000::"), 129),
			masked: Prefix{},
		},
		{
			prefix: PrefixFrom(mustIP("1.2.3.4"), 33),
			masked: Prefix{},
		},
	}
	for _, test := range tests {
		t.Run(test.prefix.String(), func(t *testing.T) {
			got := test.prefix.Masked()
			if got != test.masked {
				t.Errorf("Masked=%s, want %s", got, test.masked)
			}
		})
	}
}

func TestPrefix(t *testing.T) {
	tests := []struct {
		prefix      string
		ip          Addr
		bits        int
		str         string
		contains    []Addr
		notContains []Addr
	}{
		{
			prefix:      "192.168.0.0/24",
			ip:          mustIP("192.168.0.0"),
			bits:        24,
			contains:    mustIPs("192.168.0.1", "192.168.0.55"),
			notContains: mustIPs("192.168.1.1", "1.1.1.1"),
		},
		{
			prefix:      "192.168.1.1/32",
			ip:          mustIP("192.168.1.1"),
			bits:        32,
			contains:    mustIPs("192.168.1.1"),
			notContains: mustIPs("192.168.1.2"),
		},
		{
			prefix:      "100.64.0.0/10", // CGNAT range; prefix not multiple of 8
			ip:          mustIP("100.64.0.0"),
			bits:        10,
			contains:    mustIPs("100.64.0.0", "100.64.0.1", "100.81.251.94", "100.100.100.100", "100.127.255.254", "100.127.255.255"),
			notContains: mustIPs("100.63.255.255", "100.128.0.0"),
		},
		{
			prefix:      "2001:db8::/96",
			ip:          mustIP("2001:db8::"),
			bits:        96,
			contains:    mustIPs("2001:db8::aaaa:bbbb", "2001:db8::1"),
			notContains: mustIPs("2001:db8::1:aaaa:bbbb", "2001:db9::"),
		},
		{
			prefix:      "0.0.0.0/0",
			ip:          mustIP("0.0.0.0"),
			bits:        0,
			contains:    mustIPs("192.168.0.1", "1.1.1.1"),
			notContains: append(mustIPs("2001:db8::1"), Addr{}),
		},
		{
			prefix:      "::/0",
			ip:          mustIP("::"),
			bits:        0,
			contains:    mustIPs("::1", "2001:db8::1"),
			notContains: mustIPs("192.0.2.1"),
		},
		{
			prefix:      "2000::/3",
			ip:          mustIP("2000::"),
			bits:        3,
			contains:    mustIPs("2001:db8::1"),
			notContains: mustIPs("fe80::1"),
		},
	}
	for _, test := range tests {
		t.Run(test.prefix, func(t *testing.T) {
			prefix, err := ParsePrefix(test.prefix)
			if err != nil {
				t.Fatal(err)
			}
			if prefix.Addr() != test.ip {
				t.Errorf("IP=%s, want %s", prefix.Addr(), test.ip)
			}
			if prefix.Bits() != test.bits {
				t.Errorf("bits=%d, want %d", prefix.Bits(), test.bits)
			}
			for _, ip := range test.contains {
				if !prefix.Contains(ip) {
					t.Errorf("does not contain %s", ip)
				}
			}
			for _, ip := range test.notContains {
				if prefix.Contains(ip) {
					t.Errorf("contains %s", ip)
				}
			}
			want := test.str
			if want == "" {
				want = test.prefix
			}
			if got := prefix.String(); got != want {
				t.Errorf("prefix.String()=%q, want %q", got, want)
			}

			TestAppendToMarshal(t, prefix)
		})
	}
}

func TestPrefixFromInvalidBits(t *testing.T) {
	v4 := MustParseAddr("1.2.3.4")
	v6 := MustParseAddr("66::66")
	tests := []struct {
		ip       Addr
		in, want int
	}{
		{v4, 0, 0},
		{v6, 0, 0},
		{v4, 1, 1},
		{v4, 33, -1},
		{v6, 33, 33},
		{v6, 127, 127},
		{v6, 128, 128},
		{v4, 254, -1},
		{v4, 255, -1},
		{v4, -1, -1},
		{v6, -1, -1},
		{v4, -5, -1},
		{v6, -5, -1},
	}
	for _, tt := range tests {
		p := PrefixFrom(tt.ip, tt.in)
		if got := p.Bits(); got != tt.want {
			t.Errorf("for (%v, %v), Bits out = %v; want %v", tt.ip, tt.in, got, tt.want)
		}
	}
}

func TestParsePrefixAllocs(t *testing.T) {
	tests := []struct {
		ip    string
		slash string
	}{
		{"192.168.1.0", "/24"},
		{"aaaa:bbbb:cccc::", "/24"},
	}
	for _, test := range tests {
		prefix := test.ip + test.slash
		t.Run(prefix, func(t *testing.T) {
			ipAllocs := int(testing.AllocsPerRun(5, func() {
				ParseAddr(test.ip)
			}))
			prefixAllocs := int(testing.AllocsPerRun(5, func() {
				ParsePrefix(prefix)
			}))
			if got := prefixAllocs - ipAllocs; got != 0 {
				t.Errorf("allocs=%d, want 0", got)
			}
		})
	}
}

func TestParsePrefixError(t *testing.T) {
	tests := []struct {
		prefix string
		errstr string
	}{
		{
			prefix: "192.168.0.0",
			errstr: "no '/'",
		},
		{
			prefix: "1.257.1.1/24",
			errstr: "value >255",
		},
		{
			prefix: "1.1.1.0/q",
			errstr: "bad bits",
		},
		{
			prefix: "1.1.1.0/-1",
			errstr: "bad bits",
		},
		{
			prefix: "1.1.1.0/33",
			errstr: "out of range",
		},
		{
			prefix: "2001::/129",
			errstr: "out of range",
		},
		// Zones are not allowed: https://go.dev/issue/51899
		{
			prefix: "1.1.1.0%a/24",
			errstr: "unexpected character",
		},
		{
			prefix: "2001:db8::%a/32",
			errstr: "zones cannot be present",
		},
		{
			prefix: "1.1.1.0/+32",
			errstr: "bad bits",
		},
		{
			prefix: "1.1.1.0/-32",
			errstr: "bad bits",
		},
		{
			prefix: "1.1.1.0/032",
			errstr: "bad bits",
		},
		{
			prefix: "1.1.1.0/0032",
			errstr: "bad bits",
		},
	}
	for _, test := range tests {
		t.Run(test.prefix, func(t *testing.T) {
			_, err := ParsePrefix(test.prefix)
			if err == nil {
				t.Fatal("no error")
			}
			if got := err.Error(); !strings.Contains(got, test.errstr) {
				t.Errorf("error is missing substring %q: %s", test.errstr, got)
			}
		})
	}
}

func TestPrefixIsSingleIP(t *testing.T) {
	tests := []struct {
		ipp  Prefix
		want bool
	}{
		{ipp: mustPrefix("127.0.0.1/32"), want: true},
		{ipp: mustPrefix("127.0.0.1/31"), want: false},
		{ipp: mustPrefix("127.0.0.1/0"), want: false},
		{ipp: mustPrefix("::1/128"), want: true},
		{ipp: mustPrefix("::1/127"), want: false},
		{ipp: mustPrefix("::1/0"), want: false},
		{ipp: Prefix{}, want: false},
	}
	for _, tt := range tests {
		got := tt.ipp.IsSingleIP()
		if got != tt.want {
			t.Errorf("IsSingleIP(%v) = %v want %v", tt.ipp, got, tt.want)
		}
	}
}

func mustIPs(strs ...string) []Addr {
	var res []Addr
	for _, s := range strs {
		res = append(res, mustIP(s))
	}
	return res
}

func BenchmarkBinaryMarshalRoundTrip(b *testing.B) {
	b.ReportAllocs()
	tests := []struct {
		name string
		ip   string
	}{
		{"ipv4", "1.2.3.4"},
		{"ipv6", "2001:db8::1"},
		{"ipv6+zone", "2001:db8::1%eth0"},
	}
	for _, tc := range tests {
		b.Run(tc.name, func(b *testing.B) {
			ip := mustIP(tc.ip)
			for i := 0; i < b.N; i++ {
				bt, err := ip.MarshalBinary()
				if err != nil {
					b.Fatal(err)
				}
				var ip2 Addr
				if err := ip2.UnmarshalBinary(bt); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkStdIPv4(b *testing.B) {
	b.ReportAllocs()
	ips := []net.IP{}
	for i := 0; i < b.N; i++ {
		ip := net.IPv4(8, 8, 8, 8)
		ips = ips[:0]
		for i := 0; i < 100; i++ {
			ips = append(ips, ip)
		}
	}
}

func BenchmarkIPv4(b *testing.B) {
	b.ReportAllocs()
	ips := []Addr{}
	for i := 0; i < b.N; i++ {
		ip := IPv4(8, 8, 8, 8)
		ips = ips[:0]
		for i := 0; i < 100; i++ {
			ips = append(ips, ip)
		}
	}
}

// ip4i was one of the possible representations of IP that came up in
// discussions, inlining IPv4 addresses, but having an "overflow"
// interface for IPv6 or IPv6 + zone. This is here for benchmarking.
type ip4i struct {
	ip4    [4]byte
	flags1 byte
	flags2 byte
	flags3 byte
	flags4 byte
	ipv6   any
}

func newip4i_v4(a, b, c, d byte) ip4i {
	return ip4i{ip4: [4]byte{a, b, c, d}}
}

// BenchmarkIPv4_inline benchmarks the candidate representation, ip4i.
func BenchmarkIPv4_inline(b *testing.B) {
	b.ReportAllocs()
	ips := []ip4i{}
	for i := 0; i < b.N; i++ {
		ip := newip4i_v4(8, 8, 8, 8)
		ips = ips[:0]
		for i := 0; i < 100; i++ {
			ips = append(ips, ip)
		}
	}
}

func BenchmarkStdIPv6(b *testing.B) {
	b.ReportAllocs()
	ips := []net.IP{}
	for i := 0; i < b.N; i++ {
		ip := net.ParseIP("2001:db8::1")
		ips = ips[:0]
		for i := 0; i < 100; i++ {
			ips = append(ips, ip)
		}
	}
}

func BenchmarkIPv6(b *testing.B) {
	b.ReportAllocs()
	ips := []Addr{}
	for i := 0; i < b.N; i++ {
		ip := mustIP("2001:db8::1")
		ips = ips[:0]
		for i := 0; i < 100; i++ {
			ips = append(ips, ip)
		}
	}
}

func BenchmarkIPv4Contains(b *testing.B) {
	b.ReportAllocs()
	prefix := PrefixFrom(IPv4(192, 168, 1, 0), 24)
	ip := IPv4(192, 168, 1, 1)
	for i := 0; i < b.N; i++ {
		prefix.Contains(ip)
	}
}

func BenchmarkIPv6Contains(b *testing.B) {
	b.ReportAllocs()
	prefix := MustParsePrefix("::1/128")
	ip := MustParseAddr("::1")
	for i := 0; i < b.N; i++ {
		prefix.Contains(ip)
	}
}

var parseBenchInputs = []struct {
	name string
	ip   string
}{
	{"v4", "192.168.1.1"},
	{"v6", "fd7a:115c:a1e0:ab12:4843:cd96:626b:430b"},
	{"v6_ellipsis", "fd7a:115c::626b:430b"},
	{"v6_v4", "::ffff:192.168.140.255"},
	{"v6_zone", "1:2::ffff:192.168.140.255%eth1"},
}

func BenchmarkParseAddr(b *testing.B) {
	sinkInternValue = unique.Make(MakeAddrDetail(true, "eth1")) // Pin to not benchmark the intern package
	for _, test := range parseBenchInputs {
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sinkIP, _ = ParseAddr(test.ip)
			}
		})
	}
}

func BenchmarkStdParseIP(b *testing.B) {
	for _, test := range parseBenchInputs {
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sinkStdIP = net.ParseIP(test.ip)
			}
		})
	}
}

func BenchmarkAddrString(b *testing.B) {
	for _, test := range parseBenchInputs {
		ip := MustParseAddr(test.ip)
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sinkString = ip.String()
			}
		})
	}
}

func BenchmarkIPStringExpanded(b *testing.B) {
	for _, test := range parseBenchInputs {
		ip := MustParseAddr(test.ip)
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sinkString = ip.StringExpanded()
			}
		})
	}
}

func BenchmarkAddrMarshalText(b *testing.B) {
	for _, test := range parseBenchInputs {
		ip := MustParseAddr(test.ip)
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sinkBytes, _ = ip.MarshalText()
			}
		})
	}
}

func BenchmarkAddrPortString(b *testing.B) {
	for _, test := range parseBenchInputs {
		ip := MustParseAddr(test.ip)
		ipp := AddrPortFrom(ip, 60000)
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sinkString = ipp.String()
			}
		})
	}
}

func BenchmarkAddrPortMarshalText(b *testing.B) {
	for _, test := range parseBenchInputs {
		ip := MustParseAddr(test.ip)
		ipp := AddrPortFrom(ip, 60000)
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sinkBytes, _ = ipp.MarshalText()
			}
		})
	}
}

func BenchmarkPrefixMasking(b *testing.B) {
	tests := []struct {
		name string
		ip   Addr
		bits int
	}{
		{
			name: "IPv4 /32",
			ip:   IPv4(192, 0, 2, 0),
			bits: 32,
		},
		{
			name: "IPv4 /17",
			ip:   IPv4(192, 0, 2, 0),
			bits: 17,
		},
		{
			name: "IPv4 /0",
			ip:   IPv4(192, 0, 2, 0),
			bits: 0,
		},
		{
			name: "IPv6 /128",
			ip:   mustIP("2001:db8::1"),
			bits: 128,
		},
		{
			name: "IPv6 /65",
			ip:   mustIP("2001:db8::1"),
			bits: 65,
		},
		{
			name: "IPv6 /0",
			ip:   mustIP("2001:db8::1"),
			bits: 0,
		},
		{
			name: "IPv6 zone /128",
			ip:   mustIP("2001:db8::1%eth0"),
			bits: 128,
		},
		{
			name: "IPv6 zone /65",
			ip:   mustIP("2001:db8::1%eth0"),
			bits: 65,
		},
		{
			name: "IPv6 zone /0",
			ip:   mustIP("2001:db8::1%eth0"),
			bits: 0,
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				sinkPrefix, _ = tt.ip.Prefix(tt.bits)
			}
		})
	}
}

func BenchmarkPrefixMarshalText(b *testing.B) {
	b.ReportAllocs()
	ipp := MustParsePrefix("66.55.44.33/22")
	for i := 0; i < b.N; i++ {
		sinkBytes, _ = ipp.MarshalText()
	}
}

func BenchmarkParseAddrPort(b *testing.B) {
	for _, test := range parseBenchInputs {
		var ipp string
		if strings.HasPrefix(test.name, "v6") {
			ipp = fmt.Sprintf("[%s]:1234", test.ip)
		} else {
			ipp = fmt.Sprintf("%s:1234", test.ip)
		}
		b.Run(test.name, func(b *testing.B) {
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				sinkAddrPort, _ = ParseAddrPort(ipp)
			}
		})
	}
}

func TestAs4(t *testing.T) {
	tests := []struct {
		ip        Addr
		want      [4]byte
		wantPanic bool
	}{
		{
			ip:   mustIP("1.2.3.4"),
			want: [4]byte{1, 2, 3, 4},
		},
		{
			ip:   AddrFrom16(mustIP("1.2.3.4").As16()), // IPv4-in-IPv6
			want: [4]byte{1, 2, 3, 4},
		},
		{
			ip:   mustIP("0.0.0.0"),
			want: [4]byte{0, 0, 0, 0},
		},
		{
			ip:        Addr{},
			wantPanic: true,
		},
		{
			ip:        mustIP("::1"),
			wantPanic: true,
		},
	}
	as4 := func(ip Addr) (v [4]byte, gotPanic bool) {
		defer func() {
			if recover() != nil {
				gotPanic = true
				return
			}
		}()
		v = ip.As4()
		return
	}
	for i, tt := range tests {
		got, gotPanic := as4(tt.ip)
		if gotPanic != tt.wantPanic {
			t.Errorf("%d. panic on %v = %v; want %v", i, tt.ip, gotPanic, tt.wantPanic)
			continue
		}
		if got != tt.want {
			t.Errorf("%d. %v = %v; want %v", i, tt.ip, got, tt.want)
		}
	}
}

func TestPrefixOverlaps(t *testing.T) {
	pfx := mustPrefix
	tests := []struct {
		a, b Prefix
		want bool
	}{
		{Prefix{}, pfx("1.2.0.0/16"), false},    // first zero
		{pfx("1.2.0.0/16"), Prefix{}, false},    // second zero
		{pfx("::0/3"), pfx("0.0.0.0/3"), false}, // different families

		{pfx("1.2.0.0/16"), pfx("1.2.0.0/16"), true}, // equal

		{pfx("1.2.0.0/16"), pfx("1.2.3.0/24"), true},
		{pfx("1.2.3.0/24"), pfx("1.2.0.0/16"), true},

		{pfx("1.2.0.0/16"), pfx("1.2.3.0/32"), true},
		{pfx("1.2.3.0/32"), pfx("1.2.0.0/16"), true},

		// Match /0 either order
		{pfx("1.2.3.0/32"), pfx("0.0.0.0/0"), true},
		{pfx("0.0.0.0/0"), pfx("1.2.3.0/32"), true},

		{pfx("1.2.3.0/32"), pfx("5.5.5.5/0"), true}, // normalization not required; /0 means true

		// IPv6 overlapping
		{pfx("5::1/128"), pfx("5::0/8"), true},
		{pfx("5::0/8"), pfx("5::1/128"), true},

		// IPv6 not overlapping
		{pfx("1::1/128"), pfx("2::2/128"), false},
		{pfx("0100::0/8"), pfx("::1/128"), false},

		// IPv4-mapped IPv6 addresses should not overlap with IPv4.
		{PrefixFrom(AddrFrom16(mustIP("1.2.0.0").As16()), 16), pfx("1.2.3.0/24"), false},

		// Invalid prefixes
		{PrefixFrom(mustIP("1.2.3.4"), 33), pfx("1.2.3.0/24"), false},
		{PrefixFrom(mustIP("2000::"), 129), pfx("2000::/64"), false},
	}
	for i, tt := range tests {
		if got := tt.a.Overlaps(tt.b); got != tt.want {
			t.Errorf("%d. (%v).Overlaps(%v) = %v; want %v", i, tt.a, tt.b, got, tt.want)
		}
		// Overlaps is commutative
		if got := tt.b.Overlaps(tt.a); got != tt.want {
			t.Errorf("%d. (%v).Overlaps(%v) = %v; want %v", i, tt.b, tt.a, got, tt.want)
		}
	}
}

// Sink variables are here to force the compiler to not elide
// seemingly useless work in benchmarks and allocation tests. If you
// were to just `_ = foo()` within a test function, the compiler could
// correctly deduce that foo() does nothing and doesn't need to be
// called. By writing results to a global variable, we hide that fact
// from the compiler and force it to keep the code under test.
var (
	sinkIP          Addr
	sinkStdIP       net.IP
	sinkAddrPort    AddrPort
	sinkPrefix      Prefix
	sinkPrefixSlice []Prefix
	sinkInternValue unique.Handle[AddrDetail]
	sinkIP16        [16]byte
	sinkIP4         [4]byte
	sinkBool        bool
	sinkString      string
	sinkBytes       []byte
	sinkUDPAddr     = &net.UDPAddr{IP: make(net.IP, 0, 16)}
)

func TestNoAllocs(t *testing.T) {
	// Wrappers that panic on error, to prove that our alloc-free
	// methods are returning successfully.
	panicIP := func(ip Addr, err error) Addr {
		if err != nil {
			panic(err)
		}
		return ip
	}
	panicPfx := func(pfx Prefix, err error) Prefix {
		if err != nil {
			panic(err)
		}
		return pfx
	}
	panicIPP := func(ipp AddrPort, err error) AddrPort {
		if err != nil {
			panic(err)
		}
		return ipp
	}
	test := func(name string, f func()) {
		t.Run(name, func(t *testing.T) {
			n := testing.AllocsPerRun(1000, f)
			if n != 0 {
				t.Fatalf("allocs = %d; want 0", int(n))
			}
		})
	}

	// Addr constructors
	test("IPv4", func() { sinkIP = IPv4(1, 2, 3, 4) })
	test("AddrFrom4", func() { sinkIP = AddrFrom4([4]byte{1, 2, 3, 4}) })
	test("AddrFrom16", func() { sinkIP = AddrFrom16([16]byte{}) })
	test("ParseAddr/4", func() { sinkIP = panicIP(ParseAddr("1.2.3.4")) })
	test("ParseAddr/6", func() { sinkIP = panicIP(ParseAddr("::1")) })
	test("MustParseAddr", func() { sinkIP = MustParseAddr("1.2.3.4") })
	test("IPv6LinkLocalAllNodes", func() { sinkIP = IPv6LinkLocalAllNodes() })
	test("IPv6LinkLocalAllRouters", func() { sinkIP = IPv6LinkLocalAllRouters() })
	test("IPv6Loopback", func() { sinkIP = IPv6Loopback() })
	test("IPv6Unspecified", func() { sinkIP = IPv6Unspecified() })

	// Addr methods
	test("Addr.IsZero", func() { sinkBool = MustParseAddr("1.2.3.4").IsZero() })
	test("Addr.BitLen", func() { sinkBool = MustParseAddr("1.2.3.4").BitLen() == 8 })
	test("Addr.Zone/4", func() { sinkBool = MustParseAddr("1.2.3.4").Zone() == "" })
	test("Addr.Zone/6", func() { sinkBool = MustParseAddr("fe80::1").Zone() == "" })
	test("Addr.Zone/6zone", func() { sinkBool = MustParseAddr("fe80::1%zone").Zone() == "" })
	test("Addr.Compare", func() {
		a := MustParseAddr("1.2.3.4")
		b := MustParseAddr("2.3.4.5")
		sinkBool = a.Compare(b) == 0
	})
	test("Addr.Less", func() {
		a := MustParseAddr("1.2.3.4")
		b := MustParseAddr("2.3.4.5")
		sinkBool = a.Less(b)
	})
	test("Addr.Is4", func() { sinkBool = MustParseAddr("1.2.3.4").Is4() })
	test("Addr.Is6", func() { sinkBool = MustParseAddr("fe80::1").Is6() })
	test("Addr.Is4In6", func() { sinkBool = MustParseAddr("fe80::1").Is4In6() })
	test("Addr.Unmap", func() { sinkIP = MustParseAddr("ffff::2.3.4.5").Unmap() })
	test("Addr.WithZone", func() { sinkIP = MustParseAddr("fe80::1").WithZone("") })
	test("Addr.IsGlobalUnicast", func() { sinkBool = MustParseAddr("2001:db8::1").IsGlobalUnicast() })
	test("Addr.IsInterfaceLocalMulticast", func() { sinkBool = MustParseAddr("fe80::1").IsInterfaceLocalMulticast() })
	test("Addr.IsLinkLocalMulticast", func() { sinkBool = MustParseAddr("fe80::1").IsLinkLocalMulticast() })
	test("Addr.IsLinkLocalUnicast", func() { sinkBool = MustParseAddr("fe80::1").IsLinkLocalUnicast() })
	test("Addr.IsLoopback", func() { sinkBool = MustParseAddr("fe80::1").IsLoopback() })
	test("Addr.IsMulticast", func() { sinkBool = MustParseAddr("fe80::1").IsMulticast() })
	test("Addr.IsPrivate", func() { sinkBool = MustParseAddr("fd00::1").IsPrivate() })
	test("Addr.IsUnspecified", func() { sinkBool = IPv6Unspecified().IsUnspecified() })
	test("Addr.Prefix/4", func() { sinkPrefix = panicPfx(MustParseAddr("1.2.3.4").Prefix(20)) })
	test("Addr.Prefix/6", func() { sinkPrefix = panicPfx(MustParseAddr("fe80::1").Prefix(64)) })
	test("Addr.As16", func() { sinkIP16 = MustParseAddr("1.2.3.4").As16() })
	test("Addr.As4", func() { sinkIP4 = MustParseAddr("1.2.3.4").As4() })
	test("Addr.Next", func() { sinkIP = MustParseAddr("1.2.3.4").Next() })
	test("Addr.Prev", func() { sinkIP = MustParseAddr("1.2.3.4").Prev() })

	// AddrPort constructors
	test("AddrPortFrom", func() { sinkAddrPort = AddrPortFrom(IPv4(1, 2, 3, 4), 22) })
	test("ParseAddrPort", func() { sinkAddrPort = panicIPP(ParseAddrPort("[::1]:1234")) })
	test("MustParseAddrPort", func() { sinkAddrPort = MustParseAddrPort("[::1]:1234") })

	// Prefix constructors
	test("PrefixFrom", func() { sinkPrefix = PrefixFrom(IPv4(1, 2, 3, 4), 32) })
	test("ParsePrefix/4", func() { sinkPrefix = panicPfx(ParsePrefix("1.2.3.4/20")) })
	test("ParsePrefix/6", func() { sinkPrefix = panicPfx(ParsePrefix("fe80::1/64")) })
	test("MustParsePrefix", func() { sinkPrefix = MustParsePrefix("1.2.3.4/20") })

	// Prefix methods
	test("Prefix.Contains", func() { sinkBool = MustParsePrefix("1.2.3.0/24").Contains(MustParseAddr("1.2.3.4")) })
	test("Prefix.Overlaps", func() {
		a, b := MustParsePrefix("1.2.3.0/24"), MustParsePrefix("1.2.0.0/16")
		sinkBool = a.Overlaps(b)
	})
	test("Prefix.IsZero", func() { sinkBool = MustParsePrefix("1.2.0.0/16").IsZero() })
	test("Prefix.IsSingleIP", func() { sinkBool = MustParsePrefix("1.2.3.4/32").IsSingleIP() })
	test("Prefix.Masked", func() { sinkPrefix = MustParsePrefix("1.2.3.4/16").Masked() })
}

func TestAddrStringAllocs(t *testing.T) {
	tests := []struct {
		name       string
		ip         Addr
		wantAllocs int
	}{
		{"zero", Addr{}, 0},
		{"ipv4", MustParseAddr("192.168.1.1"), 1},
		{"ipv6", MustParseAddr("2001:db8::1"), 1},
		{"ipv6+zone", MustParseAddr("2001:db8::1%eth0"), 1},
		{"ipv4-in-ipv6", MustParseAddr("::ffff:192.168.1.1"), 1},
		{"ipv4-in-ipv6+zone", MustParseAddr("::ffff:192.168.1.1%eth0"), 1},
	}
	optimizationOff := testenv.OptimizationOff()
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if optimizationOff && strings.HasPrefix(tc.name, "ipv4-in-ipv6") {
				// Optimizations are required to remove some allocs.
				t.Skipf("skipping on %v", testenv.Builder())
			}
			allocs := int(testing.AllocsPerRun(1000, func() {
				sinkString = tc.ip.String()
			}))
			if allocs != tc.wantAllocs {
				t.Errorf("allocs=%d, want %d", allocs, tc.wantAllocs)
			}
		})
	}
}

func TestPrefixString(t *testing.T) {
	tests := []struct {
		ipp  Prefix
		want string
	}{
		{Prefix{}, "invalid Prefix"},
		{PrefixFrom(Addr{}, 8), "invalid Prefix"},
		{PrefixFrom(MustParseAddr("1.2.3.4"), 88), "invalid Prefix"},
	}

	for _, tt := range tests {
		if got := tt.ipp.String(); got != tt.want {
			t.Errorf("(%#v).String() = %q want %q", tt.ipp, got, tt.want)
		}
	}
}

func TestInvalidAddrPortString(t *testing.T) {
	tests := []struct {
		ipp  AddrPort
		want string
	}{
		{AddrPort{}, "invalid AddrPort"},
		{AddrPortFrom(Addr{}, 80), "invalid AddrPort"},
	}

	for _, tt := range tests {
		if got := tt.ipp.String(); got != tt.want {
			t.Errorf("(%#v).String() = %q want %q", tt.ipp, got, tt.want)
		}
	}
}

func TestAsSlice(t *testing.T) {
	tests := []struct {
		in   Addr
		want []byte
	}{
		{in: Addr{}, want: nil},
		{in: mustIP("1.2.3.4"), want: []byte{1, 2, 3, 4}},
		{in: mustIP("ffff::1"), want: []byte{0xff, 0xff, 15: 1}},
	}

	for _, test := range tests {
		got := test.in.AsSlice()
		if !bytes.Equal(got, test.want) {
			t.Errorf("%v.AsSlice() = %v want %v", test.in, got, test.want)
		}
	}
}

var sink16 [16]byte

func BenchmarkAs16(b *testing.B) {
	addr := MustParseAddr("1::10")
	for i := 0; i < b.N; i++ {
		sink16 = addr.As16()
	}
}
