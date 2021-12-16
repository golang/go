// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netip_test

import (
	"bytes"
	"encoding"
	"fmt"
	"net"
	. "net/netip"
	"reflect"
	"strings"
	"testing"
)

var corpus = []string{
	// Basic zero IPv4 address.
	"0.0.0.0",
	// Basic non-zero IPv4 address.
	"192.168.140.255",
	// IPv4 address in windows-style "print all the digits" form.
	"010.000.015.001",
	// IPv4 address with a silly amount of leading zeros.
	"000001.00000002.00000003.000000004",
	// 4-in-6 with octet with leading zero
	"::ffff:1.2.03.4",
	// Basic zero IPv6 address.
	"::",
	// Localhost IPv6.
	"::1",
	// Fully expanded IPv6 address.
	"fd7a:115c:a1e0:ab12:4843:cd96:626b:430b",
	// IPv6 with elided fields in the middle.
	"fd7a:115c::626b:430b",
	// IPv6 with elided fields at the end.
	"fd7a:115c:a1e0:ab12:4843:cd96::",
	// IPv6 with single elided field at the end.
	"fd7a:115c:a1e0:ab12:4843:cd96:626b::",
	"fd7a:115c:a1e0:ab12:4843:cd96:626b:0",
	// IPv6 with single elided field in the middle.
	"fd7a:115c:a1e0::4843:cd96:626b:430b",
	"fd7a:115c:a1e0:0:4843:cd96:626b:430b",
	// IPv6 with the trailing 32 bits written as IPv4 dotted decimal. (4in6)
	"::ffff:192.168.140.255",
	"::ffff:192.168.140.255",
	// IPv6 with a zone specifier.
	"fd7a:115c:a1e0:ab12:4843:cd96:626b:430b%eth0",
	// IPv6 with dotted decimal and zone specifier.
	"1:2::ffff:192.168.140.255%eth1",
	"1:2::ffff:c0a8:8cff%eth1",
	// IPv6 with capital letters.
	"FD9E:1A04:F01D::1",
	"fd9e:1a04:f01d::1",
	// Empty string.
	"",
	// Garbage non-IP.
	"bad",
	// Single number. Some parsers accept this as an IPv4 address in
	// big-endian uint32 form, but we don't.
	"1234",
	// IPv4 with a zone specifier.
	"1.2.3.4%eth0",
	// IPv4 field must have at least one digit.
	".1.2.3",
	"1.2.3.",
	"1..2.3",
	// IPv4 address too long.
	"1.2.3.4.5",
	// IPv4 in dotted octal form.
	"0300.0250.0214.0377",
	// IPv4 in dotted hex form.
	"0xc0.0xa8.0x8c.0xff",
	// IPv4 in class B form.
	"192.168.12345",
	// IPv4 in class B form, with a small enough number to be
	// parseable as a regular dotted decimal field.
	"127.0.1",
	// IPv4 in class A form.
	"192.1234567",
	// IPv4 in class A form, with a small enough number to be
	// parseable as a regular dotted decimal field.
	"127.1",
	// IPv4 field has value >255.
	"192.168.300.1",
	// IPv4 with too many fields.
	"192.168.0.1.5.6",
	// IPv6 with not enough fields.
	"1:2:3:4:5:6:7",
	// IPv6 with too many fields.
	"1:2:3:4:5:6:7:8:9",
	// IPv6 with 8 fields and a :: expander.
	"1:2:3:4::5:6:7:8",
	// IPv6 with a field bigger than 2b.
	"fe801::1",
	// IPv6 with non-hex values in field.
	"fe80:tail:scal:e::",
	// IPv6 with a zone delimiter but no zone.
	"fe80::1%",
	// IPv6 with a zone specifier of zero.
	"::ffff:0:0%0",
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
	// AddrPort strings.
	"1.2.3.4:51820",
	"[fd7a:115c:a1e0:ab12:4843:cd96:626b:430b]:80",
	"[::ffff:c000:0280]:65535",
	"[::ffff:c000:0280%eth0]:1",
	// Prefix strings.
	"1.2.3.4/24",
	"fd7a:115c:a1e0:ab12:4843:cd96:626b:430b/118",
	"::ffff:c000:0280/96",
	"::ffff:c000:0280%eth0/37",
}

func FuzzParse(f *testing.F) {
	for _, seed := range corpus {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, s string) {
		ip, _ := ParseAddr(s)
		checkStringParseRoundTrip(t, ip, ParseAddr)
		checkEncoding(t, ip)

		// Check that we match the net's IP parser, modulo zones.
		if !strings.Contains(s, "%") {
			stdip := net.ParseIP(s)
			if !ip.IsValid() != (stdip == nil) {
				t.Errorf("ParseAddr zero != net.ParseIP nil: ip=%q stdip=%q", ip, stdip)
			}

			if ip.IsValid() && !ip.Is4In6() {
				buf, err := ip.MarshalText()
				if err != nil {
					t.Fatal(err)
				}
				buf2, err := stdip.MarshalText()
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(buf, buf2) {
					t.Errorf("Addr.MarshalText() != net.IP.MarshalText(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.String() != stdip.String() {
					t.Errorf("Addr.String() != net.IP.String(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.IsGlobalUnicast() != stdip.IsGlobalUnicast() {
					t.Errorf("Addr.IsGlobalUnicast() != net.IP.IsGlobalUnicast(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.IsInterfaceLocalMulticast() != stdip.IsInterfaceLocalMulticast() {
					t.Errorf("Addr.IsInterfaceLocalMulticast() != net.IP.IsInterfaceLocalMulticast(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.IsLinkLocalMulticast() != stdip.IsLinkLocalMulticast() {
					t.Errorf("Addr.IsLinkLocalMulticast() != net.IP.IsLinkLocalMulticast(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.IsLinkLocalUnicast() != stdip.IsLinkLocalUnicast() {
					t.Errorf("Addr.IsLinkLocalUnicast() != net.IP.IsLinkLocalUnicast(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.IsLoopback() != stdip.IsLoopback() {
					t.Errorf("Addr.IsLoopback() != net.IP.IsLoopback(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.IsMulticast() != stdip.IsMulticast() {
					t.Errorf("Addr.IsMulticast() != net.IP.IsMulticast(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.IsPrivate() != stdip.IsPrivate() {
					t.Errorf("Addr.IsPrivate() != net.IP.IsPrivate(): ip=%q stdip=%q", ip, stdip)
				}
				if ip.IsUnspecified() != stdip.IsUnspecified() {
					t.Errorf("Addr.IsUnspecified() != net.IP.IsUnspecified(): ip=%q stdip=%q", ip, stdip)
				}
			}
		}

		// Check that .Next().Prev() and .Prev().Next() preserve the IP.
		if ip.IsValid() && ip.Next().IsValid() && ip.Next().Prev() != ip {
			t.Errorf(".Next.Prev did not round trip: ip=%q .next=%q .next.prev=%q", ip, ip.Next(), ip.Next().Prev())
		}
		if ip.IsValid() && ip.Prev().IsValid() && ip.Prev().Next() != ip {
			t.Errorf(".Prev.Next did not round trip: ip=%q .prev=%q .prev.next=%q", ip, ip.Prev(), ip.Prev().Next())
		}

		port, err := ParseAddrPort(s)
		if err == nil {
			checkStringParseRoundTrip(t, port, ParseAddrPort)
			checkEncoding(t, port)
		}
		port = AddrPortFrom(ip, 80)
		checkStringParseRoundTrip(t, port, ParseAddrPort)
		checkEncoding(t, port)

		ipp, err := ParsePrefix(s)
		if err == nil {
			checkStringParseRoundTrip(t, ipp, ParsePrefix)
			checkEncoding(t, ipp)
		}
		ipp = PrefixFrom(ip, 8)
		checkStringParseRoundTrip(t, ipp, ParsePrefix)
		checkEncoding(t, ipp)
	})
}

// checkTextMarshaler checks that x's MarshalText and UnmarshalText functions round trip correctly.
func checkTextMarshaler(t *testing.T, x encoding.TextMarshaler) {
	buf, err := x.MarshalText()
	if err != nil {
		t.Fatal(err)
	}
	y := reflect.New(reflect.TypeOf(x)).Interface().(encoding.TextUnmarshaler)
	err = y.UnmarshalText(buf)
	if err != nil {
		t.Logf("(%v).MarshalText() = %q", x, buf)
		t.Fatalf("(%T).UnmarshalText(%q) = %v", y, buf, err)
	}
	e := reflect.ValueOf(y).Elem().Interface()
	if !reflect.DeepEqual(x, e) {
		t.Logf("(%v).MarshalText() = %q", x, buf)
		t.Logf("(%T).UnmarshalText(%q) = %v", y, buf, y)
		t.Fatalf("MarshalText/UnmarshalText failed to round trip: %#v != %#v", x, e)
	}
	buf2, err := y.(encoding.TextMarshaler).MarshalText()
	if err != nil {
		t.Logf("(%v).MarshalText() = %q", x, buf)
		t.Logf("(%T).UnmarshalText(%q) = %v", y, buf, y)
		t.Fatalf("failed to MarshalText a second time: %v", err)
	}
	if !bytes.Equal(buf, buf2) {
		t.Logf("(%v).MarshalText() = %q", x, buf)
		t.Logf("(%T).UnmarshalText(%q) = %v", y, buf, y)
		t.Logf("(%v).MarshalText() = %q", y, buf2)
		t.Fatalf("second MarshalText differs from first: %q != %q", buf, buf2)
	}
}

// checkBinaryMarshaler checks that x's MarshalText and UnmarshalText functions round trip correctly.
func checkBinaryMarshaler(t *testing.T, x encoding.BinaryMarshaler) {
	buf, err := x.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	y := reflect.New(reflect.TypeOf(x)).Interface().(encoding.BinaryUnmarshaler)
	err = y.UnmarshalBinary(buf)
	if err != nil {
		t.Logf("(%v).MarshalBinary() = %q", x, buf)
		t.Fatalf("(%T).UnmarshalBinary(%q) = %v", y, buf, err)
	}
	e := reflect.ValueOf(y).Elem().Interface()
	if !reflect.DeepEqual(x, e) {
		t.Logf("(%v).MarshalBinary() = %q", x, buf)
		t.Logf("(%T).UnmarshalBinary(%q) = %v", y, buf, y)
		t.Fatalf("MarshalBinary/UnmarshalBinary failed to round trip: %#v != %#v", x, e)
	}
	buf2, err := y.(encoding.BinaryMarshaler).MarshalBinary()
	if err != nil {
		t.Logf("(%v).MarshalBinary() = %q", x, buf)
		t.Logf("(%T).UnmarshalBinary(%q) = %v", y, buf, y)
		t.Fatalf("failed to MarshalBinary a second time: %v", err)
	}
	if !bytes.Equal(buf, buf2) {
		t.Logf("(%v).MarshalBinary() = %q", x, buf)
		t.Logf("(%T).UnmarshalBinary(%q) = %v", y, buf, y)
		t.Logf("(%v).MarshalBinary() = %q", y, buf2)
		t.Fatalf("second MarshalBinary differs from first: %q != %q", buf, buf2)
	}
}

func checkTextMarshalMatchesString(t *testing.T, x netipType) {
	buf, err := x.MarshalText()
	if err != nil {
		t.Fatal(err)
	}
	str := x.String()
	if string(buf) != str {
		t.Fatalf("%v: MarshalText = %q, String = %q", x, buf, str)
	}
}

type appendMarshaler interface {
	encoding.TextMarshaler
	AppendTo([]byte) []byte
}

// checkTextMarshalMatchesAppendTo checks that x's MarshalText matches x's AppendTo.
func checkTextMarshalMatchesAppendTo(t *testing.T, x appendMarshaler) {
	buf, err := x.MarshalText()
	if err != nil {
		t.Fatal(err)
	}

	buf2 := make([]byte, 0, len(buf))
	buf2 = x.AppendTo(buf2)
	if !bytes.Equal(buf, buf2) {
		t.Fatalf("%v: MarshalText = %q, AppendTo = %q", x, buf, buf2)
	}
}

type netipType interface {
	encoding.BinaryMarshaler
	encoding.TextMarshaler
	fmt.Stringer
	IsValid() bool
}

type netipTypeCmp interface {
	comparable
	netipType
}

// checkStringParseRoundTrip checks that x's String method and the provided parse function can round trip correctly.
func checkStringParseRoundTrip[P netipTypeCmp](t *testing.T, x P, parse func(string) (P, error)) {
	if !x.IsValid() {
		// Ignore invalid values.
		return
	}

	s := x.String()
	y, err := parse(s)
	if err != nil {
		t.Fatalf("s=%q err=%v", s, err)
	}
	if x != y {
		t.Fatalf("%T round trip identity failure: s=%q x=%#v y=%#v", x, s, x, y)
	}
	s2 := y.String()
	if s != s2 {
		t.Fatalf("%T String round trip identity failure: s=%#v s2=%#v", x, s, s2)
	}
}

func checkEncoding(t *testing.T, x netipType) {
	if x.IsValid() {
		checkTextMarshaler(t, x)
		checkBinaryMarshaler(t, x)
		checkTextMarshalMatchesString(t, x)
	}

	if am, ok := x.(appendMarshaler); ok {
		checkTextMarshalMatchesAppendTo(t, am)
	}
}
