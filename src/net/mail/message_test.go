// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mail

import (
	"bytes"
	"io"
	"io/ioutil"
	"mime"
	"reflect"
	"strings"
	"testing"
	"time"
)

var parseTests = []struct {
	in     string
	header Header
	body   string
}{
	{
		// RFC 5322, Appendix A.1.1
		in: `From: John Doe <jdoe@machine.example>
To: Mary Smith <mary@example.net>
Subject: Saying Hello
Date: Fri, 21 Nov 1997 09:55:06 -0600
Message-ID: <1234@local.machine.example>

This is a message just to say hello.
So, "Hello".
`,
		header: Header{
			"From":       []string{"John Doe <jdoe@machine.example>"},
			"To":         []string{"Mary Smith <mary@example.net>"},
			"Subject":    []string{"Saying Hello"},
			"Date":       []string{"Fri, 21 Nov 1997 09:55:06 -0600"},
			"Message-Id": []string{"<1234@local.machine.example>"},
		},
		body: "This is a message just to say hello.\nSo, \"Hello\".\n",
	},
}

func TestParsing(t *testing.T) {
	for i, test := range parseTests {
		msg, err := ReadMessage(bytes.NewBuffer([]byte(test.in)))
		if err != nil {
			t.Errorf("test #%d: Failed parsing message: %v", i, err)
			continue
		}
		if !headerEq(msg.Header, test.header) {
			t.Errorf("test #%d: Incorrectly parsed message header.\nGot:\n%+v\nWant:\n%+v",
				i, msg.Header, test.header)
		}
		body, err := ioutil.ReadAll(msg.Body)
		if err != nil {
			t.Errorf("test #%d: Failed reading body: %v", i, err)
			continue
		}
		bodyStr := string(body)
		if bodyStr != test.body {
			t.Errorf("test #%d: Incorrectly parsed message body.\nGot:\n%+v\nWant:\n%+v",
				i, bodyStr, test.body)
		}
	}
}

func headerEq(a, b Header) bool {
	if len(a) != len(b) {
		return false
	}
	for k, as := range a {
		bs, ok := b[k]
		if !ok {
			return false
		}
		if !reflect.DeepEqual(as, bs) {
			return false
		}
	}
	return true
}

func TestDateParsing(t *testing.T) {
	tests := []struct {
		dateStr string
		exp     time.Time
	}{
		// RFC 5322, Appendix A.1.1
		{
			"Fri, 21 Nov 1997 09:55:06 -0600",
			time.Date(1997, 11, 21, 9, 55, 6, 0, time.FixedZone("", -6*60*60)),
		},
		// RFC5322, Appendix A.6.2
		// Obsolete date.
		{
			"21 Nov 97 09:55:06 GMT",
			time.Date(1997, 11, 21, 9, 55, 6, 0, time.FixedZone("GMT", 0)),
		},
		// Commonly found format not specified by RFC 5322.
		{
			"Fri, 21 Nov 1997 09:55:06 -0600 (MDT)",
			time.Date(1997, 11, 21, 9, 55, 6, 0, time.FixedZone("", -6*60*60)),
		},
	}
	for _, test := range tests {
		hdr := Header{
			"Date": []string{test.dateStr},
		}
		date, err := hdr.Date()
		if err != nil {
			t.Errorf("Failed parsing %q: %v", test.dateStr, err)
			continue
		}
		if !date.Equal(test.exp) {
			t.Errorf("Parse of %q: got %+v, want %+v", test.dateStr, date, test.exp)
		}
	}
}

func TestAddressParsingError(t *testing.T) {
	const txt = "=?iso-8859-2?Q?Bogl=E1rka_Tak=E1cs?= <unknown@gmail.com>"
	_, err := ParseAddress(txt)
	if err == nil || !strings.Contains(err.Error(), "charset not supported") {
		t.Errorf(`mail.ParseAddress(%q) err: %q, want ".*charset not supported.*"`, txt, err)
	}
}

func TestAddressParsingErrorUnquotedNonASCII(t *testing.T) {
	const txt = "µ <micro@example.net>"
	_, err := ParseAddress(txt)
	if err == nil || !strings.Contains(err.Error(), "unencoded non-ASCII text in address") {
		t.Errorf(`mail.ParseAddress(%q) err: %q, want ".*unencoded non-ASCII text in address.*"`, txt, err)
	}
}

func TestAddressParsing(t *testing.T) {
	tests := []struct {
		addrsStr string
		exp      []*Address
	}{
		// Bare address
		{
			`jdoe@machine.example`,
			[]*Address{{
				Address: "jdoe@machine.example",
			}},
		},
		// RFC 5322, Appendix A.1.1
		{
			`John Doe <jdoe@machine.example>`,
			[]*Address{{
				Name:    "John Doe",
				Address: "jdoe@machine.example",
			}},
		},
		// RFC 5322, Appendix A.1.2
		{
			`"Joe Q. Public" <john.q.public@example.com>`,
			[]*Address{{
				Name:    "Joe Q. Public",
				Address: "john.q.public@example.com",
			}},
		},
		{
			`Mary Smith <mary@x.test>, jdoe@example.org, Who? <one@y.test>`,
			[]*Address{
				{
					Name:    "Mary Smith",
					Address: "mary@x.test",
				},
				{
					Address: "jdoe@example.org",
				},
				{
					Name:    "Who?",
					Address: "one@y.test",
				},
			},
		},
		{
			`<boss@nil.test>, "Giant; \"Big\" Box" <sysservices@example.net>`,
			[]*Address{
				{
					Address: "boss@nil.test",
				},
				{
					Name:    `Giant; "Big" Box`,
					Address: "sysservices@example.net",
				},
			},
		},
		// RFC 5322, Appendix A.1.3
		// TODO(dsymonds): Group addresses.

		// RFC 2047 "Q"-encoded ISO-8859-1 address.
		{
			`=?iso-8859-1?q?J=F6rg_Doe?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jörg Doe`,
					Address: "joerg@example.com",
				},
			},
		},
		// RFC 2047 "Q"-encoded US-ASCII address. Dumb but legal.
		{
			`=?us-ascii?q?J=6Frg_Doe?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jorg Doe`,
					Address: "joerg@example.com",
				},
			},
		},
		// RFC 2047 "Q"-encoded UTF-8 address.
		{
			`=?utf-8?q?J=C3=B6rg_Doe?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jörg Doe`,
					Address: "joerg@example.com",
				},
			},
		},
		// RFC 2047, Section 8.
		{
			`=?ISO-8859-1?Q?Andr=E9?= Pirard <PIRARD@vm1.ulg.ac.be>`,
			[]*Address{
				{
					Name:    `André Pirard`,
					Address: "PIRARD@vm1.ulg.ac.be",
				},
			},
		},
		// Custom example of RFC 2047 "B"-encoded ISO-8859-1 address.
		{
			`=?ISO-8859-1?B?SvZyZw==?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jörg`,
					Address: "joerg@example.com",
				},
			},
		},
		// Custom example of RFC 2047 "B"-encoded UTF-8 address.
		{
			`=?UTF-8?B?SsO2cmc=?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jörg`,
					Address: "joerg@example.com",
				},
			},
		},
		// Custom example with "." in name. For issue 4938
		{
			`Asem H. <noreply@example.com>`,
			[]*Address{
				{
					Name:    `Asem H.`,
					Address: "noreply@example.com",
				},
			},
		},
	}
	for _, test := range tests {
		if len(test.exp) == 1 {
			addr, err := ParseAddress(test.addrsStr)
			if err != nil {
				t.Errorf("Failed parsing (single) %q: %v", test.addrsStr, err)
				continue
			}
			if !reflect.DeepEqual([]*Address{addr}, test.exp) {
				t.Errorf("Parse (single) of %q: got %+v, want %+v", test.addrsStr, addr, test.exp)
			}
		}

		addrs, err := ParseAddressList(test.addrsStr)
		if err != nil {
			t.Errorf("Failed parsing (list) %q: %v", test.addrsStr, err)
			continue
		}
		if !reflect.DeepEqual(addrs, test.exp) {
			t.Errorf("Parse (list) of %q: got %+v, want %+v", test.addrsStr, addrs, test.exp)
		}
	}
}

func TestAddressParser(t *testing.T) {
	tests := []struct {
		addrsStr string
		exp      []*Address
	}{
		// Bare address
		{
			`jdoe@machine.example`,
			[]*Address{{
				Address: "jdoe@machine.example",
			}},
		},
		// RFC 5322, Appendix A.1.1
		{
			`John Doe <jdoe@machine.example>`,
			[]*Address{{
				Name:    "John Doe",
				Address: "jdoe@machine.example",
			}},
		},
		// RFC 5322, Appendix A.1.2
		{
			`"Joe Q. Public" <john.q.public@example.com>`,
			[]*Address{{
				Name:    "Joe Q. Public",
				Address: "john.q.public@example.com",
			}},
		},
		{
			`Mary Smith <mary@x.test>, jdoe@example.org, Who? <one@y.test>`,
			[]*Address{
				{
					Name:    "Mary Smith",
					Address: "mary@x.test",
				},
				{
					Address: "jdoe@example.org",
				},
				{
					Name:    "Who?",
					Address: "one@y.test",
				},
			},
		},
		{
			`<boss@nil.test>, "Giant; \"Big\" Box" <sysservices@example.net>`,
			[]*Address{
				{
					Address: "boss@nil.test",
				},
				{
					Name:    `Giant; "Big" Box`,
					Address: "sysservices@example.net",
				},
			},
		},
		// RFC 2047 "Q"-encoded ISO-8859-1 address.
		{
			`=?iso-8859-1?q?J=F6rg_Doe?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jörg Doe`,
					Address: "joerg@example.com",
				},
			},
		},
		// RFC 2047 "Q"-encoded US-ASCII address. Dumb but legal.
		{
			`=?us-ascii?q?J=6Frg_Doe?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jorg Doe`,
					Address: "joerg@example.com",
				},
			},
		},
		// RFC 2047 "Q"-encoded ISO-8859-15 address.
		{
			`=?ISO-8859-15?Q?J=F6rg_Doe?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jörg Doe`,
					Address: "joerg@example.com",
				},
			},
		},
		// RFC 2047 "B"-encoded windows-1252 address.
		{
			`=?windows-1252?q?Andr=E9?= Pirard <PIRARD@vm1.ulg.ac.be>`,
			[]*Address{
				{
					Name:    `André Pirard`,
					Address: "PIRARD@vm1.ulg.ac.be",
				},
			},
		},
		// Custom example of RFC 2047 "B"-encoded ISO-8859-15 address.
		{
			`=?ISO-8859-15?B?SvZyZw==?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jörg`,
					Address: "joerg@example.com",
				},
			},
		},
		// Custom example of RFC 2047 "B"-encoded UTF-8 address.
		{
			`=?UTF-8?B?SsO2cmc=?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `Jörg`,
					Address: "joerg@example.com",
				},
			},
		},
		// Custom example with "." in name. For issue 4938
		{
			`Asem H. <noreply@example.com>`,
			[]*Address{
				{
					Name:    `Asem H.`,
					Address: "noreply@example.com",
				},
			},
		},
	}

	ap := AddressParser{WordDecoder: &mime.WordDecoder{
		CharsetReader: func(charset string, input io.Reader) (io.Reader, error) {
			in, err := ioutil.ReadAll(input)
			if err != nil {
				return nil, err
			}

			switch charset {
			case "iso-8859-15":
				in = bytes.Replace(in, []byte("\xf6"), []byte("ö"), -1)
			case "windows-1252":
				in = bytes.Replace(in, []byte("\xe9"), []byte("é"), -1)
			}

			return bytes.NewReader(in), nil
		},
	}}

	for _, test := range tests {
		if len(test.exp) == 1 {
			addr, err := ap.Parse(test.addrsStr)
			if err != nil {
				t.Errorf("Failed parsing (single) %q: %v", test.addrsStr, err)
				continue
			}
			if !reflect.DeepEqual([]*Address{addr}, test.exp) {
				t.Errorf("Parse (single) of %q: got %+v, want %+v", test.addrsStr, addr, test.exp)
			}
		}

		addrs, err := ap.ParseList(test.addrsStr)
		if err != nil {
			t.Errorf("Failed parsing (list) %q: %v", test.addrsStr, err)
			continue
		}
		if !reflect.DeepEqual(addrs, test.exp) {
			t.Errorf("Parse (list) of %q: got %+v, want %+v", test.addrsStr, addrs, test.exp)
		}
	}
}

func TestAddressString(t *testing.T) {
	tests := []struct {
		addr *Address
		exp  string
	}{
		{
			&Address{Address: "bob@example.com"},
			"<bob@example.com>",
		},
		{ // quoted local parts: RFC 5322, 3.4.1. and 3.2.4.
			&Address{Address: `my@idiot@address@example.com`},
			`<"my@idiot@address"@example.com>`,
		},
		{ // quoted local parts
			&Address{Address: ` @example.com`},
			`<" "@example.com>`,
		},
		{
			&Address{Name: "Bob", Address: "bob@example.com"},
			`"Bob" <bob@example.com>`,
		},
		{
			// note the ö (o with an umlaut)
			&Address{Name: "Böb", Address: "bob@example.com"},
			`=?utf-8?q?B=C3=B6b?= <bob@example.com>`,
		},
		{
			&Address{Name: "Bob Jane", Address: "bob@example.com"},
			`"Bob Jane" <bob@example.com>`,
		},
		{
			&Address{Name: "Böb Jacöb", Address: "bob@example.com"},
			`=?utf-8?q?B=C3=B6b_Jac=C3=B6b?= <bob@example.com>`,
		},
		{ // https://golang.org/issue/12098
			&Address{Name: "Rob", Address: ""},
			`"Rob" <@>`,
		},
		{ // https://golang.org/issue/12098
			&Address{Name: "Rob", Address: "@"},
			`"Rob" <@>`,
		},
		{
			&Address{Name: "Böb, Jacöb", Address: "bob@example.com"},
			`=?utf-8?b?QsO2YiwgSmFjw7Zi?= <bob@example.com>`,
		},
		{
			&Address{Name: "=??Q?x?=", Address: "hello@world.com"},
			`"=??Q?x?=" <hello@world.com>`,
		},
		{
			&Address{Name: "=?hello", Address: "hello@world.com"},
			`"=?hello" <hello@world.com>`,
		},
		{
			&Address{Name: "world?=", Address: "hello@world.com"},
			`"world?=" <hello@world.com>`,
		},
	}
	for _, test := range tests {
		s := test.addr.String()
		if s != test.exp {
			t.Errorf("Address%+v.String() = %v, want %v", *test.addr, s, test.exp)
			continue
		}

		// Check round-trip.
		if test.addr.Address != "" && test.addr.Address != "@" {
			a, err := ParseAddress(test.exp)
			if err != nil {
				t.Errorf("ParseAddress(%#q): %v", test.exp, err)
				continue
			}
			if a.Name != test.addr.Name || a.Address != test.addr.Address {
				t.Errorf("ParseAddress(%#q) = %#v, want %#v", test.exp, a, test.addr)
			}
		}
	}
}

// Check if all valid addresses can be parsed, formatted and parsed again
func TestAddressParsingAndFormatting(t *testing.T) {

	// Should pass
	tests := []string{
		`<Bob@example.com>`,
		`<bob.bob@example.com>`,
		`<".bob"@example.com>`,
		`<" "@example.com>`,
		`<some.mail-with-dash@example.com>`,
		`<"dot.and space"@example.com>`,
		`<"very.unusual.@.unusual.com"@example.com>`,
		`<admin@mailserver1>`,
		`<postmaster@localhost>`,
		"<#!$%&'*+-/=?^_`{}|~@example.org>",
		`<"very.(),:;<>[]\".VERY.\"very@\\ \"very\".unusual"@strange.example.com>`, // escaped quotes
		`<"()<>[]:,;@\\\"!#$%&'*+-/=?^_{}| ~.a"@example.org>`,                      // escaped backslashes
		`<"Abc\\@def"@example.com>`,
		`<"Joe\\Blow"@example.com>`,
		`<test1/test2=test3@example.com>`,
		`<def!xyz%abc@example.com>`,
		`<_somename@example.com>`,
		`<joe@uk>`,
		`<~@example.com>`,
		`<"..."@test.com>`,
		`<"john..doe"@example.com>`,
		`<"john.doe."@example.com>`,
		`<".john.doe"@example.com>`,
		`<"."@example.com>`,
		`<".."@example.com>`,
		`<"0:"@0>`,
	}

	for _, test := range tests {
		addr, err := ParseAddress(test)
		if err != nil {
			t.Errorf("Couldn't parse address %s: %s", test, err.Error())
			continue
		}
		str := addr.String()
		addr, err = ParseAddress(str)
		if err != nil {
			t.Errorf("ParseAddr(%q) error: %v", test, err)
			continue
		}

		if addr.String() != test {
			t.Errorf("String() round-trip = %q; want %q", addr, test)
			continue
		}

	}

	// Should fail
	badTests := []string{
		`<Abc.example.com>`,
		`<A@b@c@example.com>`,
		`<a"b(c)d,e:f;g<h>i[j\k]l@example.com>`,
		`<just"not"right@example.com>`,
		`<this is"not\allowed@example.com>`,
		`<this\ still\"not\\allowed@example.com>`,
		`<john..doe@example.com>`,
		`<john.doe@example..com>`,
		`<john.doe@example..com>`,
		`<john.doe.@example.com>`,
		`<john.doe.@.example.com>`,
		`<.john.doe@example.com>`,
		`<@example.com>`,
		`<.@example.com>`,
		`<test@.>`,
		`< @example.com>`,
		`<""test""blah""@example.com>`,
		`<""@0>`,
		"<\"\t0\"@0>",
	}

	for _, test := range badTests {
		_, err := ParseAddress(test)
		if err == nil {
			t.Errorf("Should have failed to parse address: %s", test)
			continue
		}

	}

}

func TestAddressFormattingAndParsing(t *testing.T) {
	tests := []*Address{
		{Name: "@lïce", Address: "alice@example.com"},
		{Name: "Böb O'Connor", Address: "bob@example.com"},
		{Name: "???", Address: "bob@example.com"},
		{Name: "Böb ???", Address: "bob@example.com"},
		{Name: "Böb (Jacöb)", Address: "bob@example.com"},
		{Name: "à#$%&'(),.:;<>@[]^`{|}~'", Address: "bob@example.com"},
		// https://golang.org/issue/11292
		{Name: "\"\\\x1f,\"", Address: "0@0"},
		// https://golang.org/issue/12782
		{Name: "naé, mée", Address: "test.mail@gmail.com"},
	}

	for i, test := range tests {
		parsed, err := ParseAddress(test.String())
		if err != nil {
			t.Errorf("test #%d: ParseAddr(%q) error: %v", i, test.String(), err)
			continue
		}
		if parsed.Name != test.Name {
			t.Errorf("test #%d: Parsed name = %q; want %q", i, parsed.Name, test.Name)
		}
		if parsed.Address != test.Address {
			t.Errorf("test #%d: Parsed address = %q; want %q", i, parsed.Address, test.Address)
		}
	}
}
