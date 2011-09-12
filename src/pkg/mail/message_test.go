// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mail

import (
	"bytes"
	"io/ioutil"
	"reflect"
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
		exp     *time.Time
	}{
		// RFC 5322, Appendix A.1.1
		{
			"Fri, 21 Nov 1997 09:55:06 -0600",
			&time.Time{
				Year:       1997,
				Month:      11,
				Day:        21,
				Hour:       9,
				Minute:     55,
				Second:     6,
				ZoneOffset: -6 * 60 * 60,
			},
		},
		// RFC5322, Appendix A.6.2
		// Obsolete date.
		{
			"21 Nov 97 09:55:06 GMT",
			&time.Time{
				Year:   1997,
				Month:  11,
				Day:    21,
				Hour:   9,
				Minute: 55,
				Second: 6,
				Zone:   "GMT",
			},
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
		if !reflect.DeepEqual(date, test.exp) {
			t.Errorf("Parse of %q: got %+v, want %+v", test.dateStr, date, test.exp)
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
			[]*Address{&Address{
				Address: "jdoe@machine.example",
			}},
		},
		// RFC 5322, Appendix A.1.1
		{
			`John Doe <jdoe@machine.example>`,
			[]*Address{&Address{
				Name:    "John Doe",
				Address: "jdoe@machine.example",
			}},
		},
		// RFC 5322, Appendix A.1.2
		{
			`"Joe Q. Public" <john.q.public@example.com>`,
			[]*Address{&Address{
				Name:    "Joe Q. Public",
				Address: "john.q.public@example.com",
			}},
		},
		{
			`Mary Smith <mary@x.test>, jdoe@example.org, Who? <one@y.test>`,
			[]*Address{
				&Address{
					Name:    "Mary Smith",
					Address: "mary@x.test",
				},
				&Address{
					Address: "jdoe@example.org",
				},
				&Address{
					Name:    "Who?",
					Address: "one@y.test",
				},
			},
		},
		{
			`<boss@nil.test>, "Giant; \"Big\" Box" <sysservices@example.net>`,
			[]*Address{
				&Address{
					Address: "boss@nil.test",
				},
				&Address{
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
				&Address{
					Name:    `Jörg Doe`,
					Address: "joerg@example.com",
				},
			},
		},
		// RFC 2047 "Q"-encoded UTF-8 address.
		{
			`=?utf-8?q?J=C3=B6rg_Doe?= <joerg@example.com>`,
			[]*Address{
				&Address{
					Name:    `Jörg Doe`,
					Address: "joerg@example.com",
				},
			},
		},
		// RFC 2047, Section 8.
		{
			`=?ISO-8859-1?Q?Andr=E9?= Pirard <PIRARD@vm1.ulg.ac.be>`,
			[]*Address{
				&Address{
					Name:    `André Pirard`,
					Address: "PIRARD@vm1.ulg.ac.be",
				},
			},
		},
		// Custom example of RFC 2047 "B"-encoded ISO-8859-1 address.
		{
			`=?ISO-8859-1?B?SvZyZw==?= <joerg@example.com>`,
			[]*Address{
				&Address{
					Name:    `Jörg`,
					Address: "joerg@example.com",
				},
			},
		},
		// Custom example of RFC 2047 "B"-encoded UTF-8 address.
		{
			`=?UTF-8?B?SsO2cmc=?= <joerg@example.com>`,
			[]*Address{
				&Address{
					Name:    `Jörg`,
					Address: "joerg@example.com",
				},
			},
		},
	}
	for _, test := range tests {
		addrs, err := newAddrParser(test.addrsStr).parseAddressList()
		if err != nil {
			t.Errorf("Failed parsing %q: %v", test.addrsStr, err)
			continue
		}
		if !reflect.DeepEqual(addrs, test.exp) {
			t.Errorf("Parse of %q: got %+v, want %+v", test.addrsStr, addrs, test.exp)
		}
	}
}

func TestAddressFormatting(t *testing.T) {
	tests := []struct {
		addr *Address
		exp  string
	}{
		{
			&Address{Address: "bob@example.com"},
			"<bob@example.com>",
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
	}
	for _, test := range tests {
		s := test.addr.String()
		if s != test.exp {
			t.Errorf("Address%+v.String() = %v, want %v", *test.addr, s, test.exp)
		}
	}
}
