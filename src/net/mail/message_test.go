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
		// RFC 5322, Appendix A.6.2
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
			t.Errorf("Header(Date: %s).Date(): %v", test.dateStr, err)
		} else if !date.Equal(test.exp) {
			t.Errorf("Header(Date: %s).Date() = %+v, want %+v", test.dateStr, date, test.exp)
		}

		date, err = ParseDate(test.dateStr)
		if err != nil {
			t.Errorf("ParseDate(%s): %v", test.dateStr, err)
		} else if !date.Equal(test.exp) {
			t.Errorf("ParseDate(%s) = %+v, want %+v", test.dateStr, date, test.exp)
		}
	}
}

func TestAddressParsingError(t *testing.T) {
	mustErrTestCases := [...]struct {
		text        string
		wantErrText string
	}{
		0:  {"=?iso-8859-2?Q?Bogl=E1rka_Tak=E1cs?= <unknown@gmail.com>", "charset not supported"},
		1:  {"a@gmail.com b@gmail.com", "expected single address"},
		2:  {string([]byte{0xed, 0xa0, 0x80}) + " <micro@example.net>", "invalid utf-8 in address"},
		3:  {"\"" + string([]byte{0xed, 0xa0, 0x80}) + "\" <half-surrogate@example.com>", "invalid utf-8 in quoted-string"},
		4:  {"\"\\" + string([]byte{0x80}) + "\" <escaped-invalid-unicode@example.net>", "invalid utf-8 in quoted-string"},
		5:  {"\"\x00\" <null@example.net>", "bad character in quoted-string"},
		6:  {"\"\\\x00\" <escaped-null@example.net>", "bad character in quoted-string"},
		7:  {"John Doe", "no angle-addr"},
		8:  {`<jdoe#machine.example>`, "missing @ in addr-spec"},
		9:  {`John <middle> Doe <jdoe@machine.example>`, "missing @ in addr-spec"},
		10: {"cfws@example.com (", "misformatted parenthetical comment"},
		11: {"empty group: ;", "empty group"},
		12: {"root group: embed group: null@example.com;", "no angle-addr"},
		13: {"group not closed: null@example.com", "expected comma"},
		14: {"group: first@example.com, second@example.com;", "group with multiple addresses"},
		15: {"john.doe", "missing '@' or angle-addr"},
		16: {"john.doe@", "no angle-addr"},
		17: {"John Doe@foo.bar", "no angle-addr"},
	}

	for i, tc := range mustErrTestCases {
		_, err := ParseAddress(tc.text)
		if err == nil || !strings.Contains(err.Error(), tc.wantErrText) {
			t.Errorf(`mail.ParseAddress(%q) #%d want %q, got %v`, tc.text, i, tc.wantErrText, err)
		}
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
			`"John (middle) Doe" <jdoe@machine.example>`,
			[]*Address{{
				Name:    "John (middle) Doe",
				Address: "jdoe@machine.example",
			}},
		},
		{
			`John (middle) Doe <jdoe@machine.example>`,
			[]*Address{{
				Name:    "John (middle) Doe",
				Address: "jdoe@machine.example",
			}},
		},
		{
			`John !@M@! Doe <jdoe@machine.example>`,
			[]*Address{{
				Name:    "John !@M@! Doe",
				Address: "jdoe@machine.example",
			}},
		},
		{
			`"John <middle> Doe" <jdoe@machine.example>`,
			[]*Address{{
				Name:    "John <middle> Doe",
				Address: "jdoe@machine.example",
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
		// RFC 5322, Appendix A.6.1
		{
			`Joe Q. Public <john.q.public@example.com>`,
			[]*Address{{
				Name:    "Joe Q. Public",
				Address: "john.q.public@example.com",
			}},
		},
		// RFC 5322, Appendix A.1.3
		{
			`group1: groupaddr1@example.com;`,
			[]*Address{
				{
					Name:    "",
					Address: "groupaddr1@example.com",
				},
			},
		},
		{
			`empty group: ;`,
			[]*Address(nil),
		},
		{
			`A Group:Ed Jones <c@a.test>,joe@where.test,John <jdoe@one.test>;`,
			[]*Address{
				{
					Name:    "Ed Jones",
					Address: "c@a.test",
				},
				{
					Name:    "",
					Address: "joe@where.test",
				},
				{
					Name:    "John",
					Address: "jdoe@one.test",
				},
			},
		},
		{
			`Group1: <addr1@example.com>;, Group 2: addr2@example.com;, John <addr3@example.com>`,
			[]*Address{
				{
					Name:    "",
					Address: "addr1@example.com",
				},
				{
					Name:    "",
					Address: "addr2@example.com",
				},
				{
					Name:    "John",
					Address: "addr3@example.com",
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
		// RFC 2047 "Q"-encoded UTF-8 address with multiple encoded-words.
		{
			`=?utf-8?q?J=C3=B6rg?=  =?utf-8?q?Doe?= <joerg@example.com>`,
			[]*Address{
				{
					Name:    `JörgDoe`,
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
		// RFC 6532 3.2.3, qtext /= UTF8-non-ascii
		{
			`"Gø Pher" <gopher@example.com>`,
			[]*Address{
				{
					Name:    `Gø Pher`,
					Address: "gopher@example.com",
				},
			},
		},
		// RFC 6532 3.2, atext /= UTF8-non-ascii
		{
			`µ <micro@example.com>`,
			[]*Address{
				{
					Name:    `µ`,
					Address: "micro@example.com",
				},
			},
		},
		// RFC 6532 3.2.2, local address parts allow UTF-8
		{
			`Micro <µ@example.com>`,
			[]*Address{
				{
					Name:    `Micro`,
					Address: "µ@example.com",
				},
			},
		},
		// RFC 6532 3.2.4, domains parts allow UTF-8
		{
			`Micro <micro@µ.example.com>`,
			[]*Address{
				{
					Name:    `Micro`,
					Address: "micro@µ.example.com",
				},
			},
		},
		// Issue 14866
		{
			`"" <emptystring@example.com>`,
			[]*Address{
				{
					Name:    "",
					Address: "emptystring@example.com",
				},
			},
		},
		// CFWS
		{
			`<cfws@example.com> (CFWS (cfws))  (another comment)`,
			[]*Address{
				{
					Name:    "",
					Address: "cfws@example.com",
				},
			},
		},
		{
			`<cfws@example.com> ()  (another comment), <cfws2@example.com> (another)`,
			[]*Address{
				{
					Name:    "",
					Address: "cfws@example.com",
				},
				{
					Name:    "",
					Address: "cfws2@example.com",
				},
			},
		},
		// Comment as display name
		{
			`john@example.com (John Doe)`,
			[]*Address{
				{
					Name:    "John Doe",
					Address: "john@example.com",
				},
			},
		},
		// Comment and display name
		{
			`John Doe <john@example.com> (Joey)`,
			[]*Address{
				{
					Name:    "John Doe",
					Address: "john@example.com",
				},
			},
		},
		// Comment as display name, no space
		{
			`john@example.com(John Doe)`,
			[]*Address{
				{
					Name:    "John Doe",
					Address: "john@example.com",
				},
			},
		},
		// Comment as display name, Q-encoded
		{
			`asjo@example.com (Adam =?utf-8?Q?Sj=C3=B8gren?=)`,
			[]*Address{
				{
					Name:    "Adam Sjøgren",
					Address: "asjo@example.com",
				},
			},
		},
		// Comment as display name, Q-encoded and tab-separated
		{
			`asjo@example.com (Adam	=?utf-8?Q?Sj=C3=B8gren?=)`,
			[]*Address{
				{
					Name:    "Adam Sjøgren",
					Address: "asjo@example.com",
				},
			},
		},
		// Nested comment as display name, Q-encoded
		{
			`asjo@example.com (Adam =?utf-8?Q?Sj=C3=B8gren?= (Debian))`,
			[]*Address{
				{
					Name:    "Adam Sjøgren (Debian)",
					Address: "asjo@example.com",
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
				in = bytes.ReplaceAll(in, []byte("\xf6"), []byte("ö"))
			case "windows-1252":
				in = bytes.ReplaceAll(in, []byte("\xe9"), []byte("é"))
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
		{
			// should q-encode even for invalid utf-8.
			&Address{Name: string([]byte{0xed, 0xa0, 0x80}), Address: "invalid-utf8@example.net"},
			"=?utf-8?q?=ED=A0=80?= <invalid-utf8@example.net>",
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
