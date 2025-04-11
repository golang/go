// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import (
	"bytes"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

func indentNewlines(s string) string {
	return strings.Join(strings.Split(s, "\n"), "\n\t")
}

func stripWhitespace(s string) string {
	return strings.Map(func(r rune) rune {
		if r == ' ' || r == '\n' || r == '\r' || r == '\t' {
			return -1
		}
		return r
	}, s)
}

func TestValid(t *testing.T) {
	tests := []struct {
		CaseName
		data string
		ok   bool
	}{
		{Name(""), `foo`, false},
		{Name(""), `}{`, false},
		{Name(""), `{]`, false},
		{Name(""), `{}`, true},
		{Name(""), `{"foo":"bar"}`, true},
		{Name(""), `{"foo":"bar","bar":{"baz":["qux"]}}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			if ok := Valid([]byte(tt.data)); ok != tt.ok {
				t.Errorf("%s: Valid(`%s`) = %v, want %v", tt.Where, tt.data, ok, tt.ok)
			}
		})
	}
}

func TestCompactAndIndent(t *testing.T) {
	tests := []struct {
		CaseName
		compact string
		indent  string
	}{
		{Name(""), `1`, `1`},
		{Name(""), `{}`, `{}`},
		{Name(""), `[]`, `[]`},
		{Name(""), `{"":2}`, "{\n\t\"\": 2\n}"},
		{Name(""), `[3]`, "[\n\t3\n]"},
		{Name(""), `[1,2,3]`, "[\n\t1,\n\t2,\n\t3\n]"},
		{Name(""), `{"x":1}`, "{\n\t\"x\": 1\n}"},
		{Name(""), `[true,false,null,"x",1,1.5,0,-5e+2]`, `[
	true,
	false,
	null,
	"x",
	1,
	1.5,
	0,
	-5e+2
]`},
		{Name(""), "{\"\":\"<>&\u2028\u2029\"}", "{\n\t\"\": \"<>&\u2028\u2029\"\n}"}, // See golang.org/issue/34070
	}
	var buf bytes.Buffer
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			buf.Reset()
			if err := Compact(&buf, []byte(tt.compact)); err != nil {
				t.Errorf("%s: Compact error: %v", tt.Where, err)
			} else if got := buf.String(); got != tt.compact {
				t.Errorf("%s: Compact:\n\tgot:  %s\n\twant: %s", tt.Where, indentNewlines(got), indentNewlines(tt.compact))
			}

			buf.Reset()
			if err := Compact(&buf, []byte(tt.indent)); err != nil {
				t.Errorf("%s: Compact error: %v", tt.Where, err)
			} else if got := buf.String(); got != tt.compact {
				t.Errorf("%s: Compact:\n\tgot:  %s\n\twant: %s", tt.Where, indentNewlines(got), indentNewlines(tt.compact))
			}

			buf.Reset()
			if err := Indent(&buf, []byte(tt.indent), "", "\t"); err != nil {
				t.Errorf("%s: Indent error: %v", tt.Where, err)
			} else if got := buf.String(); got != tt.indent {
				t.Errorf("%s: Compact:\n\tgot:  %s\n\twant: %s", tt.Where, indentNewlines(got), indentNewlines(tt.indent))
			}

			buf.Reset()
			if err := Indent(&buf, []byte(tt.compact), "", "\t"); err != nil {
				t.Errorf("%s: Indent error: %v", tt.Where, err)
			} else if got := buf.String(); got != tt.indent {
				t.Errorf("%s: Compact:\n\tgot:  %s\n\twant: %s", tt.Where, indentNewlines(got), indentNewlines(tt.indent))
			}
		})
	}
}

func TestCompactSeparators(t *testing.T) {
	// U+2028 and U+2029 should be escaped inside strings.
	// They should not appear outside strings.
	tests := []struct {
		CaseName
		in, compact string
	}{
		{Name(""), "{\"\u2028\": 1}", "{\"\u2028\":1}"},
		{Name(""), "{\"\u2029\" :2}", "{\"\u2029\":2}"},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := Compact(&buf, []byte(tt.in)); err != nil {
				t.Errorf("%s: Compact error: %v", tt.Where, err)
			} else if got := buf.String(); got != tt.compact {
				t.Errorf("%s: Compact:\n\tgot:  %s\n\twant: %s", tt.Where, indentNewlines(got), indentNewlines(tt.compact))
			}
		})
	}
}

// Tests of a large random structure.

func TestCompactBig(t *testing.T) {
	initBig()
	var buf bytes.Buffer
	if err := Compact(&buf, jsonBig); err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	b := buf.Bytes()
	if !bytes.Equal(b, jsonBig) {
		t.Error("Compact:")
		diff(t, b, jsonBig)
		return
	}
}

func TestIndentBig(t *testing.T) {
	t.Parallel()
	initBig()
	var buf bytes.Buffer
	if err := Indent(&buf, jsonBig, "", "\t"); err != nil {
		t.Fatalf("Indent error: %v", err)
	}
	b := buf.Bytes()
	if len(b) == len(jsonBig) {
		// jsonBig is compact (no unnecessary spaces);
		// indenting should make it bigger
		t.Fatalf("Indent did not expand the input")
	}

	// should be idempotent
	var buf1 bytes.Buffer
	if err := Indent(&buf1, b, "", "\t"); err != nil {
		t.Fatalf("Indent error: %v", err)
	}
	b1 := buf1.Bytes()
	if !bytes.Equal(b1, b) {
		t.Error("Indent(Indent(jsonBig)) != Indent(jsonBig):")
		diff(t, b1, b)
		return
	}

	// should get back to original
	buf1.Reset()
	if err := Compact(&buf1, b); err != nil {
		t.Fatalf("Compact error: %v", err)
	}
	b1 = buf1.Bytes()
	if !bytes.Equal(b1, jsonBig) {
		t.Error("Compact(Indent(jsonBig)) != jsonBig:")
		diff(t, b1, jsonBig)
		return
	}
}

func TestIndentErrors(t *testing.T) {
	tests := []struct {
		CaseName
		in  string
		err error
	}{
		{Name(""), `{"X": "foo", "Y"}`, &SyntaxError{"invalid character '}' after object key", 17}},
		{Name(""), `{"X": "foo" "Y": "bar"}`, &SyntaxError{"invalid character '\"' after object key:value pair", 13}},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			slice := make([]uint8, 0)
			buf := bytes.NewBuffer(slice)
			if err := Indent(buf, []uint8(tt.in), "", ""); err != nil {
				if !reflect.DeepEqual(err, tt.err) {
					t.Fatalf("%s: Indent error:\n\tgot:  %v\n\twant: %v", tt.Where, err, tt.err)
				}
			}
		})
	}
}

func diff(t *testing.T, a, b []byte) {
	t.Helper()
	for i := 0; ; i++ {
		if i >= len(a) || i >= len(b) || a[i] != b[i] {
			j := i - 10
			if j < 0 {
				j = 0
			}
			t.Errorf("diverge at %d: «%s» vs «%s»", i, trim(a[j:]), trim(b[j:]))
			return
		}
	}
}

func trim(b []byte) []byte {
	return b[:min(len(b), 20)]
}

// Generate a random JSON object.

var jsonBig []byte

func initBig() {
	n := 10000
	if testing.Short() {
		n = 100
	}
	b, err := Marshal(genValue(n))
	if err != nil {
		panic(err)
	}
	jsonBig = b
}

func genValue(n int) any {
	if n > 1 {
		switch rand.Intn(2) {
		case 0:
			return genArray(n)
		case 1:
			return genMap(n)
		}
	}
	switch rand.Intn(3) {
	case 0:
		return rand.Intn(2) == 0
	case 1:
		return rand.NormFloat64()
	case 2:
		return genString(30)
	}
	panic("unreachable")
}

func genString(stddev float64) string {
	n := int(math.Abs(rand.NormFloat64()*stddev + stddev/2))
	c := make([]rune, n)
	for i := range c {
		f := math.Abs(rand.NormFloat64()*64 + 32)
		if f > 0x10ffff {
			f = 0x10ffff
		}
		c[i] = rune(f)
	}
	return string(c)
}

func genArray(n int) []any {
	f := int(math.Abs(rand.NormFloat64()) * math.Min(10, float64(n/2)))
	if f > n {
		f = n
	}
	if f < 1 {
		f = 1
	}
	x := make([]any, f)
	for i := range x {
		x[i] = genValue(((i+1)*n)/f - (i*n)/f)
	}
	return x
}

func genMap(n int) map[string]any {
	f := int(math.Abs(rand.NormFloat64()) * math.Min(10, float64(n/2)))
	if f > n {
		f = n
	}
	if n > 0 && f == 0 {
		f = 1
	}
	x := make(map[string]any)
	for i := 0; i < f; i++ {
		x[genString(10)] = genValue(((i+1)*n)/f - (i*n)/f)
	}
	return x
}
