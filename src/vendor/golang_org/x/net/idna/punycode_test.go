// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package idna

import (
	"strings"
	"testing"
)

var punycodeTestCases = [...]struct {
	s, encoded string
}{
	{"", ""},
	{"-", "--"},
	{"-a", "-a-"},
	{"-a-", "-a--"},
	{"a", "a-"},
	{"a-", "a--"},
	{"a-b", "a-b-"},
	{"books", "books-"},
	{"bücher", "bcher-kva"},
	{"Hello世界", "Hello-ck1hg65u"},
	{"ü", "tda"},
	{"üý", "tdac"},

	// The test cases below come from RFC 3492 section 7.1 with Errata 3026.
	{
		// (A) Arabic (Egyptian).
		"\u0644\u064A\u0647\u0645\u0627\u0628\u062A\u0643\u0644" +
			"\u0645\u0648\u0634\u0639\u0631\u0628\u064A\u061F",
		"egbpdaj6bu4bxfgehfvwxn",
	},
	{
		// (B) Chinese (simplified).
		"\u4ED6\u4EEC\u4E3A\u4EC0\u4E48\u4E0D\u8BF4\u4E2D\u6587",
		"ihqwcrb4cv8a8dqg056pqjye",
	},
	{
		// (C) Chinese (traditional).
		"\u4ED6\u5011\u7232\u4EC0\u9EBD\u4E0D\u8AAA\u4E2D\u6587",
		"ihqwctvzc91f659drss3x8bo0yb",
	},
	{
		// (D) Czech.
		"\u0050\u0072\u006F\u010D\u0070\u0072\u006F\u0073\u0074" +
			"\u011B\u006E\u0065\u006D\u006C\u0075\u0076\u00ED\u010D" +
			"\u0065\u0073\u006B\u0079",
		"Proprostnemluvesky-uyb24dma41a",
	},
	{
		// (E) Hebrew.
		"\u05DC\u05DE\u05D4\u05D4\u05DD\u05E4\u05E9\u05D5\u05D8" +
			"\u05DC\u05D0\u05DE\u05D3\u05D1\u05E8\u05D9\u05DD\u05E2" +
			"\u05D1\u05E8\u05D9\u05EA",
		"4dbcagdahymbxekheh6e0a7fei0b",
	},
	{
		// (F) Hindi (Devanagari).
		"\u092F\u0939\u0932\u094B\u0917\u0939\u093F\u0928\u094D" +
			"\u0926\u0940\u0915\u094D\u092F\u094B\u0902\u0928\u0939" +
			"\u0940\u0902\u092C\u094B\u0932\u0938\u0915\u0924\u0947" +
			"\u0939\u0948\u0902",
		"i1baa7eci9glrd9b2ae1bj0hfcgg6iyaf8o0a1dig0cd",
	},
	{
		// (G) Japanese (kanji and hiragana).
		"\u306A\u305C\u307F\u3093\u306A\u65E5\u672C\u8A9E\u3092" +
			"\u8A71\u3057\u3066\u304F\u308C\u306A\u3044\u306E\u304B",
		"n8jok5ay5dzabd5bym9f0cm5685rrjetr6pdxa",
	},
	{
		// (H) Korean (Hangul syllables).
		"\uC138\uACC4\uC758\uBAA8\uB4E0\uC0AC\uB78C\uB4E4\uC774" +
			"\uD55C\uAD6D\uC5B4\uB97C\uC774\uD574\uD55C\uB2E4\uBA74" +
			"\uC5BC\uB9C8\uB098\uC88B\uC744\uAE4C",
		"989aomsvi5e83db1d2a355cv1e0vak1dwrv93d5xbh15a0dt30a5j" +
			"psd879ccm6fea98c",
	},
	{
		// (I) Russian (Cyrillic).
		"\u043F\u043E\u0447\u0435\u043C\u0443\u0436\u0435\u043E" +
			"\u043D\u0438\u043D\u0435\u0433\u043E\u0432\u043E\u0440" +
			"\u044F\u0442\u043F\u043E\u0440\u0443\u0441\u0441\u043A" +
			"\u0438",
		"b1abfaaepdrnnbgefbadotcwatmq2g4l",
	},
	{
		// (J) Spanish.
		"\u0050\u006F\u0072\u0071\u0075\u00E9\u006E\u006F\u0070" +
			"\u0075\u0065\u0064\u0065\u006E\u0073\u0069\u006D\u0070" +
			"\u006C\u0065\u006D\u0065\u006E\u0074\u0065\u0068\u0061" +
			"\u0062\u006C\u0061\u0072\u0065\u006E\u0045\u0073\u0070" +
			"\u0061\u00F1\u006F\u006C",
		"PorqunopuedensimplementehablarenEspaol-fmd56a",
	},
	{
		// (K) Vietnamese.
		"\u0054\u1EA1\u0069\u0073\u0061\u006F\u0068\u1ECD\u006B" +
			"\u0068\u00F4\u006E\u0067\u0074\u0068\u1EC3\u0063\u0068" +
			"\u1EC9\u006E\u00F3\u0069\u0074\u0069\u1EBF\u006E\u0067" +
			"\u0056\u0069\u1EC7\u0074",
		"TisaohkhngthchnitingVit-kjcr8268qyxafd2f1b9g",
	},
	{
		// (L) 3<nen>B<gumi><kinpachi><sensei>.
		"\u0033\u5E74\u0042\u7D44\u91D1\u516B\u5148\u751F",
		"3B-ww4c5e180e575a65lsy2b",
	},
	{
		// (M) <amuro><namie>-with-SUPER-MONKEYS.
		"\u5B89\u5BA4\u5948\u7F8E\u6075\u002D\u0077\u0069\u0074" +
			"\u0068\u002D\u0053\u0055\u0050\u0045\u0052\u002D\u004D" +
			"\u004F\u004E\u004B\u0045\u0059\u0053",
		"-with-SUPER-MONKEYS-pc58ag80a8qai00g7n9n",
	},
	{
		// (N) Hello-Another-Way-<sorezore><no><basho>.
		"\u0048\u0065\u006C\u006C\u006F\u002D\u0041\u006E\u006F" +
			"\u0074\u0068\u0065\u0072\u002D\u0057\u0061\u0079\u002D" +
			"\u305D\u308C\u305E\u308C\u306E\u5834\u6240",
		"Hello-Another-Way--fc4qua05auwb3674vfr0b",
	},
	{
		// (O) <hitotsu><yane><no><shita>2.
		"\u3072\u3068\u3064\u5C4B\u6839\u306E\u4E0B\u0032",
		"2-u9tlzr9756bt3uc0v",
	},
	{
		// (P) Maji<de>Koi<suru>5<byou><mae>
		"\u004D\u0061\u006A\u0069\u3067\u004B\u006F\u0069\u3059" +
			"\u308B\u0035\u79D2\u524D",
		"MajiKoi5-783gue6qz075azm5e",
	},
	{
		// (Q) <pafii>de<runba>
		"\u30D1\u30D5\u30A3\u30FC\u0064\u0065\u30EB\u30F3\u30D0",
		"de-jg4avhby1noc0d",
	},
	{
		// (R) <sono><supiido><de>
		"\u305D\u306E\u30B9\u30D4\u30FC\u30C9\u3067",
		"d9juau41awczczp",
	},
	{
		// (S) -> $1.00 <-
		"\u002D\u003E\u0020\u0024\u0031\u002E\u0030\u0030\u0020" +
			"\u003C\u002D",
		"-> $1.00 <--",
	},
}

func TestPunycode(t *testing.T) {
	for _, tc := range punycodeTestCases {
		if got, err := decode(tc.encoded); err != nil {
			t.Errorf("decode(%q): %v", tc.encoded, err)
		} else if got != tc.s {
			t.Errorf("decode(%q): got %q, want %q", tc.encoded, got, tc.s)
		}

		if got, err := encode("", tc.s); err != nil {
			t.Errorf(`encode("", %q): %v`, tc.s, err)
		} else if got != tc.encoded {
			t.Errorf(`encode("", %q): got %q, want %q`, tc.s, got, tc.encoded)
		}
	}
}

var punycodeErrorTestCases = [...]string{
	"decode -",            // A sole '-' is invalid.
	"decode foo\x00bar",   // '\x00' is not in [0-9A-Za-z].
	"decode foo#bar",      // '#' is not in [0-9A-Za-z].
	"decode foo\u00A3bar", // '\u00A3' is not in [0-9A-Za-z].
	"decode 9",            // "9a" decodes to codepoint \u00A3; "9" is truncated.
	"decode 99999a",       // "99999a" decodes to codepoint \U0048A3C1, which is > \U0010FFFF.
	"decode 9999999999a",  // "9999999999a" overflows the int32 calculation.

	"encode " + strings.Repeat("x", 65536) + "\uff00", // int32 overflow.
}

func TestPunycodeErrors(t *testing.T) {
	for _, tc := range punycodeErrorTestCases {
		var err error
		switch {
		case strings.HasPrefix(tc, "decode "):
			_, err = decode(tc[7:])
		case strings.HasPrefix(tc, "encode "):
			_, err = encode("", tc[7:])
		}
		if err == nil {
			if len(tc) > 256 {
				tc = tc[:100] + "..." + tc[len(tc)-100:]
			}
			t.Errorf("no error for %s", tc)
		}
	}
}
