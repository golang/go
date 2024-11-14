// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netip

import (
	"bytes"
	"encoding"
	"encoding/json"
	"strings"
	"testing"
)

var (
	mustPrefix = MustParsePrefix
	mustIP     = MustParseAddr
)

func TestPrefixValid(t *testing.T) {
	v4 := MustParseAddr("1.2.3.4")
	v6 := MustParseAddr("::1")
	tests := []struct {
		ipp  Prefix
		want bool
	}{
		{PrefixFrom(v4, -2), false},
		{PrefixFrom(v4, -1), false},
		{PrefixFrom(v4, 0), true},
		{PrefixFrom(v4, 32), true},
		{PrefixFrom(v4, 33), false},

		{PrefixFrom(v6, -2), false},
		{PrefixFrom(v6, -1), false},
		{PrefixFrom(v6, 0), true},
		{PrefixFrom(v6, 32), true},
		{PrefixFrom(v6, 128), true},
		{PrefixFrom(v6, 129), false},

		{PrefixFrom(Addr{}, -2), false},
		{PrefixFrom(Addr{}, -1), false},
		{PrefixFrom(Addr{}, 0), false},
		{PrefixFrom(Addr{}, 32), false},
		{PrefixFrom(Addr{}, 128), false},
	}
	for _, tt := range tests {
		got := tt.ipp.IsValid()
		if got != tt.want {
			t.Errorf("(%v).IsValid() = %v want %v", tt.ipp, got, tt.want)
		}

		// Test that there is only one invalid Prefix representation per Addr.
		invalid := PrefixFrom(tt.ipp.Addr(), -1)
		if !got && tt.ipp != invalid {
			t.Errorf("(%v == %v) = false, want true", tt.ipp, invalid)
		}
	}
}

var nextPrevTests = []struct {
	ip   Addr
	next Addr
	prev Addr
}{
	{mustIP("10.0.0.1"), mustIP("10.0.0.2"), mustIP("10.0.0.0")},
	{mustIP("10.0.0.255"), mustIP("10.0.1.0"), mustIP("10.0.0.254")},
	{mustIP("127.0.0.1"), mustIP("127.0.0.2"), mustIP("127.0.0.0")},
	{mustIP("254.255.255.255"), mustIP("255.0.0.0"), mustIP("254.255.255.254")},
	{mustIP("255.255.255.255"), Addr{}, mustIP("255.255.255.254")},
	{mustIP("0.0.0.0"), mustIP("0.0.0.1"), Addr{}},
	{mustIP("::"), mustIP("::1"), Addr{}},
	{mustIP("::%x"), mustIP("::1%x"), Addr{}},
	{mustIP("::1"), mustIP("::2"), mustIP("::")},
	{mustIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"), Addr{}, mustIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe")},
}

func TestIPNextPrev(t *testing.T) {
	doNextPrev(t)

	for _, ip := range []Addr{
		mustIP("0.0.0.0"),
		mustIP("::"),
	} {
		got := ip.Prev()
		if !got.isZero() {
			t.Errorf("IP(%v).Prev = %v; want zero", ip, got)
		}
	}

	var allFF [16]byte
	for i := range allFF {
		allFF[i] = 0xff
	}

	for _, ip := range []Addr{
		mustIP("255.255.255.255"),
		AddrFrom16(allFF),
	} {
		got := ip.Next()
		if !got.isZero() {
			t.Errorf("IP(%v).Next = %v; want zero", ip, got)
		}
	}
}

func BenchmarkIPNextPrev(b *testing.B) {
	for i := 0; i < b.N; i++ {
		doNextPrev(b)
	}
}

func doNextPrev(t testing.TB) {
	for _, tt := range nextPrevTests {
		gnext, gprev := tt.ip.Next(), tt.ip.Prev()
		if gnext != tt.next {
			t.Errorf("IP(%v).Next = %v; want %v", tt.ip, gnext, tt.next)
		}
		if gprev != tt.prev {
			t.Errorf("IP(%v).Prev = %v; want %v", tt.ip, gprev, tt.prev)
		}
		if !tt.ip.Next().isZero() && tt.ip.Next().Prev() != tt.ip {
			t.Errorf("IP(%v).Next.Prev = %v; want %v", tt.ip, tt.ip.Next().Prev(), tt.ip)
		}
		if !tt.ip.Prev().isZero() && tt.ip.Prev().Next() != tt.ip {
			t.Errorf("IP(%v).Prev.Next = %v; want %v", tt.ip, tt.ip.Prev().Next(), tt.ip)
		}
	}
}

func TestIPBitLen(t *testing.T) {
	tests := []struct {
		ip   Addr
		want int
	}{
		{Addr{}, 0},
		{mustIP("0.0.0.0"), 32},
		{mustIP("10.0.0.1"), 32},
		{mustIP("::"), 128},
		{mustIP("fed0::1"), 128},
		{mustIP("::ffff:10.0.0.1"), 128},
	}
	for _, tt := range tests {
		got := tt.ip.BitLen()
		if got != tt.want {
			t.Errorf("BitLen(%v) = %d; want %d", tt.ip, got, tt.want)
		}
	}
}

func TestPrefixContains(t *testing.T) {
	tests := []struct {
		ipp  Prefix
		ip   Addr
		want bool
	}{
		{mustPrefix("9.8.7.6/0"), mustIP("9.8.7.6"), true},
		{mustPrefix("9.8.7.6/16"), mustIP("9.8.7.6"), true},
		{mustPrefix("9.8.7.6/16"), mustIP("9.8.6.4"), true},
		{mustPrefix("9.8.7.6/16"), mustIP("9.9.7.6"), false},
		{mustPrefix("9.8.7.6/32"), mustIP("9.8.7.6"), true},
		{mustPrefix("9.8.7.6/32"), mustIP("9.8.7.7"), false},
		{mustPrefix("9.8.7.6/32"), mustIP("9.8.7.7"), false},
		{mustPrefix("::1/0"), mustIP("::1"), true},
		{mustPrefix("::1/0"), mustIP("::2"), true},
		{mustPrefix("::1/127"), mustIP("::1"), true},
		{mustPrefix("::1/127"), mustIP("::2"), false},
		{mustPrefix("::1/128"), mustIP("::1"), true},
		{mustPrefix("::1/127"), mustIP("::2"), false},
		// Zones ignored: https://go.dev/issue/51899
		{Prefix{mustIP("1.2.3.4").WithZone("a"), 32}, mustIP("1.2.3.4"), true},
		{Prefix{mustIP("::1").WithZone("a"), 128}, mustIP("::1"), true},
		// invalid IP
		{mustPrefix("::1/0"), Addr{}, false},
		{mustPrefix("1.2.3.4/0"), Addr{}, false},
		// invalid Prefix
		{PrefixFrom(mustIP("::1"), 129), mustIP("::1"), false},
		{PrefixFrom(mustIP("1.2.3.4"), 33), mustIP("1.2.3.4"), false},
		{PrefixFrom(Addr{}, 0), mustIP("1.2.3.4"), false},
		{PrefixFrom(Addr{}, 32), mustIP("1.2.3.4"), false},
		{PrefixFrom(Addr{}, 128), mustIP("::1"), false},
		// wrong IP family
		{mustPrefix("::1/0"), mustIP("1.2.3.4"), false},
		{mustPrefix("1.2.3.4/0"), mustIP("::1"), false},
	}
	for _, tt := range tests {
		got := tt.ipp.Contains(tt.ip)
		if got != tt.want {
			t.Errorf("(%v).Contains(%v) = %v want %v", tt.ipp, tt.ip, got, tt.want)
		}
	}
}

func TestParseIPError(t *testing.T) {
	tests := []struct {
		ip     string
		errstr string
	}{
		{
			ip: "localhost",
		},
		{
			ip:     "500.0.0.1",
			errstr: "field has value >255",
		},
		{
			ip:     "::gggg%eth0",
			errstr: "must have at least one digit",
		},
		{
			ip:     "fe80::1cc0:3e8c:119f:c2e1%",
			errstr: "zone must be a non-empty string",
		},
		{
			ip:     "%eth0",
			errstr: "missing IPv6 address",
		},
	}
	for _, test := range tests {
		t.Run(test.ip, func { t ->
			_, err := ParseAddr(test.ip)
			if err == nil {
				t.Fatal("no error")
			}
			if _, ok := err.(parseAddrError); !ok {
				t.Errorf("error type is %T, want parseIPError", err)
			}
			if test.errstr == "" {
				test.errstr = "unable to parse IP"
			}
			if got := err.Error(); !strings.Contains(got, test.errstr) {
				t.Errorf("error is missing substring %q: %s", test.errstr, got)
			}
		})
	}
}

func TestParseAddrPort(t *testing.T) {
	tests := []struct {
		in      string
		want    AddrPort
		wantErr bool
	}{
		{in: "1.2.3.4:1234", want: AddrPort{mustIP("1.2.3.4"), 1234}},
		{in: "1.1.1.1:123456", wantErr: true},
		{in: "1.1.1.1:-123", wantErr: true},
		{in: "[::1]:1234", want: AddrPort{mustIP("::1"), 1234}},
		{in: "[1.2.3.4]:1234", wantErr: true},
		{in: "fe80::1:1234", wantErr: true},
		{in: ":0", wantErr: true}, // if we need to parse this form, there should be a separate function that explicitly allows it
	}
	for _, test := range tests {
		t.Run(test.in, func { t ->
			got, err := ParseAddrPort(test.in)
			if err != nil {
				if test.wantErr {
					return
				}
				t.Fatal(err)
			}
			if got != test.want {
				t.Errorf("got %v; want %v", got, test.want)
			}
			if got.String() != test.in {
				t.Errorf("String = %q; want %q", got.String(), test.in)
			}
		})

		t.Run(test.in+"/AppendTo", func { t ->
			got, err := ParseAddrPort(test.in)
			if err == nil {
				testAppendToMarshal(t, got)
			}
		})

		// TextMarshal and TextUnmarshal mostly behave like
		// ParseAddrPort and String. Divergent behavior are handled in
		// TestAddrPortMarshalUnmarshal.
		t.Run(test.in+"/Marshal", func { t ->
			var got AddrPort
			jsin := `"` + test.in + `"`
			err := json.Unmarshal([]byte(jsin), &got)
			if err != nil {
				if test.wantErr {
					return
				}
				t.Fatal(err)
			}
			if got != test.want {
				t.Errorf("got %v; want %v", got, test.want)
			}
			gotb, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if string(gotb) != jsin {
				t.Errorf("Marshal = %q; want %q", string(gotb), jsin)
			}
		})
	}
}

func TestAddrPortMarshalUnmarshal(t *testing.T) {
	tests := []struct {
		in   string
		want AddrPort
	}{
		{"", AddrPort{}},
	}

	for _, test := range tests {
		t.Run(test.in, func { t ->
			orig := `"` + test.in + `"`

			var ipp AddrPort
			if err := json.Unmarshal([]byte(orig), &ipp); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}

			ippb, err := json.Marshal(ipp)
			if err != nil {
				t.Fatalf("failed to marshal: %v", err)
			}

			back := string(ippb)
			if orig != back {
				t.Errorf("Marshal = %q; want %q", back, orig)
			}

			testAppendToMarshal(t, ipp)
		})
	}
}

type appendMarshaler interface {
	encoding.TextMarshaler
	AppendTo([]byte) []byte
}

// testAppendToMarshal tests that x's AppendTo and MarshalText methods yield the same results.
// x's MarshalText method must not return an error.
func testAppendToMarshal(t *testing.T, x appendMarshaler) {
	t.Helper()
	m, err := x.MarshalText()
	if err != nil {
		t.Fatalf("(%v).MarshalText: %v", x, err)
	}
	a := make([]byte, 0, len(m))
	a = x.AppendTo(a)
	if !bytes.Equal(m, a) {
		t.Errorf("(%v).MarshalText = %q, (%v).AppendTo = %q", x, m, x, a)
	}
}

func TestIPv6Accessor(t *testing.T) {
	var a [16]byte
	for i := range a {
		a[i] = uint8(i) + 1
	}
	ip := AddrFrom16(a)
	for i := range a {
		if got, want := ip.v6(uint8(i)), uint8(i)+1; got != want {
			t.Errorf("v6(%v) = %v; want %v", i, got, want)
		}
	}
}
