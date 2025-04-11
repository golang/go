// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"io"
	"strings"
	"testing"

	"encoding/json/internal/jsontest"
	"encoding/json/internal/jsonwire"
)

type valueTestdataEntry struct {
	name                jsontest.CaseName
	in                  string
	wantValid           bool
	wantCompacted       string
	wantCompactErr      error  // implies wantCompacted is in
	wantIndented        string // wantCompacted if empty; uses "\t" for indent prefix and "    " for indent
	wantIndentErr       error  // implies wantCompacted is in
	wantCanonicalized   string // wantCompacted if empty
	wantCanonicalizeErr error  // implies wantCompacted is in
}

var valueTestdata = append(func() (out []valueTestdataEntry) {
	// Initialize valueTestdata from coderTestdata.
	for _, td := range coderTestdata {
		// NOTE: The Compact method preserves the raw formatting of strings,
		// while the Encoder (by default) does not.
		if td.name.Name == "ComplicatedString" {
			td.outCompacted = strings.TrimSpace(td.in)
		}
		out = append(out, valueTestdataEntry{
			name:              td.name,
			in:                td.in,
			wantValid:         true,
			wantCompacted:     td.outCompacted,
			wantIndented:      td.outIndented,
			wantCanonicalized: td.outCanonicalized,
		})
	}
	return out
}(), []valueTestdataEntry{{
	name: jsontest.Name("RFC8785/Primitives"),
	in: `{
		"numbers": [333333333.33333329, 1E30, 4.50,
					2e-3, 0.000000000000000000000000001, -0],
		"string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/",
		"literals": [null, true, false]
	}`,
	wantValid:     true,
	wantCompacted: `{"numbers":[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001,-0],"string":"\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/","literals":[null,true,false]}`,
	wantIndented: `{
	    "numbers": [
	        333333333.33333329,
	        1E30,
	        4.50,
	        2e-3,
	        0.000000000000000000000000001,
	        -0
	    ],
	    "string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/",
	    "literals": [
	        null,
	        true,
	        false
	    ]
	}`,
	wantCanonicalized: `{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27,0],"string":"€$\u000f\nA'B\"\\\\\"/"}`,
}, {
	name: jsontest.Name("RFC8785/ObjectOrdering"),
	in: `{
		"\u20ac": "Euro Sign",
		"\r": "Carriage Return",
		"\ufb33": "Hebrew Letter Dalet With Dagesh",
		"1": "One",
		"\ud83d\ude00": "Emoji: Grinning Face",
		"\u0080": "Control",
		"\u00f6": "Latin Small Letter O With Diaeresis"
	}`,
	wantValid:     true,
	wantCompacted: `{"\u20ac":"Euro Sign","\r":"Carriage Return","\ufb33":"Hebrew Letter Dalet With Dagesh","1":"One","\ud83d\ude00":"Emoji: Grinning Face","\u0080":"Control","\u00f6":"Latin Small Letter O With Diaeresis"}`,
	wantIndented: `{
	    "\u20ac": "Euro Sign",
	    "\r": "Carriage Return",
	    "\ufb33": "Hebrew Letter Dalet With Dagesh",
	    "1": "One",
	    "\ud83d\ude00": "Emoji: Grinning Face",
	    "\u0080": "Control",
	    "\u00f6": "Latin Small Letter O With Diaeresis"
	}`,
	wantCanonicalized: `{"\r":"Carriage Return","1":"One","":"Control","ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign","😀":"Emoji: Grinning Face","דּ":"Hebrew Letter Dalet With Dagesh"}`,
}, {
	name:          jsontest.Name("LargeIntegers"),
	in:            ` [ -9223372036854775808 , 9223372036854775807 ] `,
	wantValid:     true,
	wantCompacted: `[-9223372036854775808,9223372036854775807]`,
	wantIndented: `[
	    -9223372036854775808,
	    9223372036854775807
	]`,
	wantCanonicalized: `[-9223372036854776000,9223372036854776000]`, // NOTE: Loss of precision due to numbers being treated as floats.
}, {
	name:                jsontest.Name("InvalidUTF8"),
	in:                  `  "living` + "\xde\xad\xbe\xef" + `\ufffd�"  `,
	wantValid:           false, // uses RFC 7493 as the definition; which validates UTF-8
	wantCompacted:       `"living` + "\xde\xad\xbe\xef" + `\ufffd�"`,
	wantCanonicalizeErr: E(jsonwire.ErrInvalidUTF8).withPos(`  "living`+"\xde\xad", ""),
}, {
	name:                jsontest.Name("InvalidUTF8/SurrogateHalf"),
	in:                  `"\ud800"`,
	wantValid:           false, // uses RFC 7493 as the definition; which validates UTF-8
	wantCompacted:       `"\ud800"`,
	wantCanonicalizeErr: newInvalidEscapeSequenceError(`\ud800"`).withPos(`"`, ""),
}, {
	name:              jsontest.Name("UppercaseEscaped"),
	in:                `"\u000B"`,
	wantValid:         true,
	wantCompacted:     `"\u000B"`,
	wantCanonicalized: `"\u000b"`,
}, {
	name:          jsontest.Name("DuplicateNames"),
	in:            ` { "0" : 0 , "1" : 1 , "0" : 0 }`,
	wantValid:     false, // uses RFC 7493 as the definition; which does check for object uniqueness
	wantCompacted: `{"0":0,"1":1,"0":0}`,
	wantIndented: `{
	    "0": 0,
	    "1": 1,
	    "0": 0
	}`,
	wantCanonicalizeErr: E(ErrDuplicateName).withPos(` { "0" : 0 , "1" : 1 , `, "/0"),
}, {
	name:                jsontest.Name("Whitespace"),
	in:                  " \n\r\t",
	wantValid:           false,
	wantCompacted:       " \n\r\t",
	wantCompactErr:      E(io.ErrUnexpectedEOF).withPos(" \n\r\t", ""),
	wantIndentErr:       E(io.ErrUnexpectedEOF).withPos(" \n\r\t", ""),
	wantCanonicalizeErr: E(io.ErrUnexpectedEOF).withPos(" \n\r\t", ""),
}}...)

func TestValueMethods(t *testing.T) {
	for _, td := range valueTestdata {
		t.Run(td.name.Name, func(t *testing.T) {
			if td.wantIndented == "" {
				td.wantIndented = td.wantCompacted
			}
			if td.wantCanonicalized == "" {
				td.wantCanonicalized = td.wantCompacted
			}
			if td.wantCompactErr != nil {
				td.wantCompacted = td.in
			}
			if td.wantIndentErr != nil {
				td.wantIndented = td.in
			}
			if td.wantCanonicalizeErr != nil {
				td.wantCanonicalized = td.in
			}

			v := Value(td.in)
			gotValid := v.IsValid()
			if gotValid != td.wantValid {
				t.Errorf("%s: Value.IsValid = %v, want %v", td.name.Where, gotValid, td.wantValid)
			}

			gotCompacted := Value(td.in)
			gotCompactErr := gotCompacted.Compact()
			if string(gotCompacted) != td.wantCompacted {
				t.Errorf("%s: Value.Compact = %s, want %s", td.name.Where, gotCompacted, td.wantCompacted)
			}
			if !equalError(gotCompactErr, td.wantCompactErr) {
				t.Errorf("%s: Value.Compact error mismatch:\ngot  %v\nwant %v", td.name.Where, gotCompactErr, td.wantCompactErr)
			}

			gotIndented := Value(td.in)
			gotIndentErr := gotIndented.Indent(WithIndentPrefix("\t"), WithIndent("    "))
			if string(gotIndented) != td.wantIndented {
				t.Errorf("%s: Value.Indent = %s, want %s", td.name.Where, gotIndented, td.wantIndented)
			}
			if !equalError(gotIndentErr, td.wantIndentErr) {
				t.Errorf("%s: Value.Indent error mismatch:\ngot  %v\nwant %v", td.name.Where, gotIndentErr, td.wantIndentErr)
			}

			gotCanonicalized := Value(td.in)
			gotCanonicalizeErr := gotCanonicalized.Canonicalize()
			if string(gotCanonicalized) != td.wantCanonicalized {
				t.Errorf("%s: Value.Canonicalize = %s, want %s", td.name.Where, gotCanonicalized, td.wantCanonicalized)
			}
			if !equalError(gotCanonicalizeErr, td.wantCanonicalizeErr) {
				t.Errorf("%s: Value.Canonicalize error mismatch:\ngot  %v\nwant %v", td.name.Where, gotCanonicalizeErr, td.wantCanonicalizeErr)
			}
		})
	}
}
