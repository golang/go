// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"errors"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

func TestEncodeWord(t *testing.T) {
	utf8, iso88591 := "utf-8", "iso-8859-1"
	tests := []struct {
		enc      WordEncoder
		charset  string
		src, exp string
	}{
		{QEncoding, utf8, "François-Jérôme", "=?utf-8?q?Fran=C3=A7ois-J=C3=A9r=C3=B4me?="},
		{BEncoding, utf8, "Café", "=?utf-8?b?Q2Fmw6k=?="},
		{QEncoding, iso88591, "La Seleção", "=?iso-8859-1?q?La_Sele=C3=A7=C3=A3o?="},
		{QEncoding, utf8, "", ""},
		{QEncoding, utf8, "A", "A"},
		{QEncoding, iso88591, "a", "a"},
		{QEncoding, utf8, "123 456", "123 456"},
		{QEncoding, utf8, "\t !\"#$%&'()*+,-./ :;<>?@[\\]^_`{|}~", "\t !\"#$%&'()*+,-./ :;<>?@[\\]^_`{|}~"},
		{QEncoding, utf8, strings.Repeat("é", 10), "=?utf-8?q?" + strings.Repeat("=C3=A9", 10) + "?="},
		{QEncoding, utf8, strings.Repeat("é", 11), "=?utf-8?q?" + strings.Repeat("=C3=A9", 10) + "?= =?utf-8?q?=C3=A9?="},
		{QEncoding, iso88591, strings.Repeat("\xe9", 22), "=?iso-8859-1?q?" + strings.Repeat("=E9", 22) + "?="},
		{QEncoding, utf8, strings.Repeat("\x80", 22), "=?utf-8?q?" + strings.Repeat("=80", 21) + "?= =?utf-8?q?=80?="},
		{BEncoding, utf8, strings.Repeat("é", 24), "=?utf-8?b?" + strings.Repeat("w6nDqcOp", 8) + "?="},
		{BEncoding, utf8, strings.Repeat("é", 27), "=?utf-8?b?" + strings.Repeat("w6nDqcOp", 8) + "?= =?utf-8?b?w6nDqcOp?="},
		{BEncoding, iso88591, strings.Repeat("\xe9", 45), "=?iso-8859-1?b?" + strings.Repeat("6enp", 15) + "?="},
		{BEncoding, utf8, strings.Repeat("\x80", 51), "=?utf-8?b?" + strings.Repeat("gICA", 16) + "?= =?utf-8?b?gICA?="},
	}

	for _, test := range tests {
		if s := test.enc.Encode(test.charset, test.src); s != test.exp {
			t.Errorf("Encode(%q) = %q, want %q", test.src, s, test.exp)
		}
	}
}

func TestDecodeWord(t *testing.T) {
	tests := []struct {
		src, exp string
		hasErr   bool
	}{
		{"=?UTF-8?Q?=C2=A1Hola,_se=C3=B1or!?=", "¡Hola, señor!", false},
		{"=?UTF-8?Q?Fran=C3=A7ois-J=C3=A9r=C3=B4me?=", "François-Jérôme", false},
		{"=?UTF-8?q?ascii?=", "ascii", false},
		{"=?utf-8?B?QW5kcsOp?=", "André", false},
		{"=?ISO-8859-1?Q?Rapha=EBl_Dupont?=", "Raphaël Dupont", false},
		{"=?utf-8?b?IkFudG9uaW8gSm9zw6kiIDxqb3NlQGV4YW1wbGUub3JnPg==?=", `"Antonio José" <jose@example.org>`, false},
		{"=?UTF-8?A?Test?=", "", true},
		{"=?UTF-8?Q?A=B?=", "", true},
		{"=?UTF-8?Q?=A?=", "", true},
		{"=?UTF-8?A?A?=", "", true},
	}

	for _, test := range tests {
		dec := new(WordDecoder)
		s, err := dec.Decode(test.src)
		if test.hasErr && err == nil {
			t.Errorf("Decode(%q) should return an error", test.src)
			continue
		}
		if !test.hasErr && err != nil {
			t.Errorf("Decode(%q): %v", test.src, err)
			continue
		}
		if s != test.exp {
			t.Errorf("Decode(%q) = %q, want %q", test.src, s, test.exp)
		}
	}
}

func TestDecodeHeader(t *testing.T) {
	tests := []struct {
		src, exp string
	}{
		{"=?UTF-8?Q?=C2=A1Hola,_se=C3=B1or!?=", "¡Hola, señor!"},
		{"=?UTF-8?Q?Fran=C3=A7ois-J=C3=A9r=C3=B4me?=", "François-Jérôme"},
		{"=?UTF-8?q?ascii?=", "ascii"},
		{"=?utf-8?B?QW5kcsOp?=", "André"},
		{"=?ISO-8859-1?Q?Rapha=EBl_Dupont?=", "Raphaël Dupont"},
		{"Jean", "Jean"},
		{"=?utf-8?b?IkFudG9uaW8gSm9zw6kiIDxqb3NlQGV4YW1wbGUub3JnPg==?=", `"Antonio José" <jose@example.org>`},
		{"=?UTF-8?A?Test?=", "=?UTF-8?A?Test?="},
		{"=?UTF-8?Q?A=B?=", "=?UTF-8?Q?A=B?="},
		{"=?UTF-8?Q?=A?=", "=?UTF-8?Q?=A?="},
		{"=?UTF-8?A?A?=", "=?UTF-8?A?A?="},
		// Incomplete words
		{"=?", "=?"},
		{"=?UTF-8?", "=?UTF-8?"},
		{"=?UTF-8?=", "=?UTF-8?="},
		{"=?UTF-8?Q", "=?UTF-8?Q"},
		{"=?UTF-8?Q?", "=?UTF-8?Q?"},
		{"=?UTF-8?Q?=", "=?UTF-8?Q?="},
		{"=?UTF-8?Q?A", "=?UTF-8?Q?A"},
		{"=?UTF-8?Q?A?", "=?UTF-8?Q?A?"},
		// Tests from RFC 2047
		{"=?ISO-8859-1?Q?a?=", "a"},
		{"=?ISO-8859-1?Q?a?= b", "a b"},
		{"=?ISO-8859-1?Q?a?= =?ISO-8859-1?Q?b?=", "ab"},
		{"=?ISO-8859-1?Q?a?=  =?ISO-8859-1?Q?b?=", "ab"},
		{"=?ISO-8859-1?Q?a?= \r\n\t =?ISO-8859-1?Q?b?=", "ab"},
		{"=?ISO-8859-1?Q?a_b?=", "a b"},
	}

	for _, test := range tests {
		dec := new(WordDecoder)
		s, err := dec.DecodeHeader(test.src)
		if err != nil {
			t.Errorf("DecodeHeader(%q): %v", test.src, err)
		}
		if s != test.exp {
			t.Errorf("DecodeHeader(%q) = %q, want %q", test.src, s, test.exp)
		}
	}
}

func TestCharsetDecoder(t *testing.T) {
	tests := []struct {
		src      string
		want     string
		charsets []string
		content  []string
	}{
		{"=?utf-8?b?Q2Fmw6k=?=", "Café", nil, nil},
		{"=?ISO-8859-1?Q?caf=E9?=", "café", nil, nil},
		{"=?US-ASCII?Q?foo_bar?=", "foo bar", nil, nil},
		{"=?utf-8?Q?=?=", "=?utf-8?Q?=?=", nil, nil},
		{"=?utf-8?Q?=A?=", "=?utf-8?Q?=A?=", nil, nil},
		{
			"=?ISO-8859-15?Q?f=F5=F6?=  =?windows-1252?Q?b=E0r?=",
			"f\xf5\xf6b\xe0r",
			[]string{"iso-8859-15", "windows-1252"},
			[]string{"f\xf5\xf6", "b\xe0r"},
		},
	}

	for _, test := range tests {
		i := 0
		dec := &WordDecoder{
			CharsetReader: func(charset string, input io.Reader) (io.Reader, error) {
				if charset != test.charsets[i] {
					t.Errorf("DecodeHeader(%q), got charset %q, want %q", test.src, charset, test.charsets[i])
				}
				content, err := ioutil.ReadAll(input)
				if err != nil {
					t.Errorf("DecodeHeader(%q), error in reader: %v", test.src, err)
				}
				got := string(content)
				if got != test.content[i] {
					t.Errorf("DecodeHeader(%q), got content %q, want %q", test.src, got, test.content[i])
				}
				i++

				return strings.NewReader(got), nil
			},
		}
		got, err := dec.DecodeHeader(test.src)
		if err != nil {
			t.Errorf("DecodeHeader(%q): %v", test.src, err)
		}
		if got != test.want {
			t.Errorf("DecodeHeader(%q) = %q, want %q", test.src, got, test.want)
		}
	}
}

func TestCharsetDecoderError(t *testing.T) {
	dec := &WordDecoder{
		CharsetReader: func(charset string, input io.Reader) (io.Reader, error) {
			return nil, errors.New("Test error")
		},
	}

	if _, err := dec.DecodeHeader("=?charset?Q?foo?="); err == nil {
		t.Error("DecodeHeader should return an error")
	}
}

func BenchmarkQEncodeWord(b *testing.B) {
	for i := 0; i < b.N; i++ {
		QEncoding.Encode("UTF-8", "¡Hola, señor!")
	}
}

func BenchmarkQDecodeWord(b *testing.B) {
	dec := new(WordDecoder)

	for i := 0; i < b.N; i++ {
		dec.Decode("=?utf-8?q?=C2=A1Hola,_se=C3=B1or!?=")
	}
}

func BenchmarkQDecodeHeader(b *testing.B) {
	dec := new(WordDecoder)

	for i := 0; i < b.N; i++ {
		dec.Decode("=?utf-8?q?=C2=A1Hola,_se=C3=B1or!?=")
	}
}
