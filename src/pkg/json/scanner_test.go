// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"math"
	"rand"
	"testing"
)

// Tests of simple examples.

type example struct {
	compact string
	indent  string
}

var examples = []example{
	{`1`, `1`},
	{`{}`, `{}`},
	{`[]`, `[]`},
	{`{"":2}`, "{\n\t\"\": 2\n}"},
	{`[3]`, "[\n\t3\n]"},
	{`[1,2,3]`, "[\n\t1,\n\t2,\n\t3\n]"},
	{`{"x":1}`, "{\n\t\"x\": 1\n}"},
	{ex1, ex1i},
}

var ex1 = `[true,false,null,"x",1,1.5,0,-5e+2]`

var ex1i = `[
	true,
	false,
	null,
	"x",
	1,
	1.5,
	0,
	-5e+2
]`

func TestCompact(t *testing.T) {
	var buf bytes.Buffer
	for _, tt := range examples {
		buf.Reset()
		if err := Compact(&buf, []byte(tt.compact)); err != nil {
			t.Errorf("Compact(%#q): %v", tt.compact, err)
		} else if s := buf.String(); s != tt.compact {
			t.Errorf("Compact(%#q) = %#q, want original", tt.compact, s)
		}

		buf.Reset()
		if err := Compact(&buf, []byte(tt.indent)); err != nil {
			t.Errorf("Compact(%#q): %v", tt.indent, err)
			continue
		} else if s := buf.String(); s != tt.compact {
			t.Errorf("Compact(%#q) = %#q, want %#q", tt.indent, s, tt.compact)
		}
	}
}

func TestIndent(t *testing.T) {
	var buf bytes.Buffer
	for _, tt := range examples {
		buf.Reset()
		if err := Indent(&buf, []byte(tt.indent), "", "\t"); err != nil {
			t.Errorf("Indent(%#q): %v", tt.indent, err)
		} else if s := buf.String(); s != tt.indent {
			t.Errorf("Indent(%#q) = %#q, want original", tt.indent, s)
		}

		buf.Reset()
		if err := Indent(&buf, []byte(tt.compact), "", "\t"); err != nil {
			t.Errorf("Indent(%#q): %v", tt.compact, err)
			continue
		} else if s := buf.String(); s != tt.indent {
			t.Errorf("Indent(%#q) = %#q, want %#q", tt.compact, s, tt.indent)
		}
	}
}

// Tests of a large random structure.

func TestCompactBig(t *testing.T) {
	var buf bytes.Buffer
	if err := Compact(&buf, jsonBig); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	b := buf.Bytes()
	if bytes.Compare(b, jsonBig) != 0 {
		t.Error("Compact(jsonBig) != jsonBig")
		diff(t, b, jsonBig)
		return
	}
}

func TestIndentBig(t *testing.T) {
	var buf bytes.Buffer
	if err := Indent(&buf, jsonBig, "", "\t"); err != nil {
		t.Fatalf("Indent1: %v", err)
	}
	b := buf.Bytes()
	if len(b) == len(jsonBig) {
		// jsonBig is compact (no unnecessary spaces);
		// indenting should make it bigger
		t.Fatalf("Indent(jsonBig) did not get bigger")
	}

	// should be idempotent
	var buf1 bytes.Buffer
	if err := Indent(&buf1, b, "", "\t"); err != nil {
		t.Fatalf("Indent2: %v", err)
	}
	b1 := buf1.Bytes()
	if bytes.Compare(b1, b) != 0 {
		t.Error("Indent(Indent(jsonBig)) != Indent(jsonBig)")
		diff(t, b1, b)
		return
	}

	// should get back to original
	buf1.Reset()
	if err := Compact(&buf1, b); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	b1 = buf1.Bytes()
	if bytes.Compare(b1, jsonBig) != 0 {
		t.Error("Compact(Indent(jsonBig)) != jsonBig")
		diff(t, b1, jsonBig)
		return
	}
}

func TestNextValueBig(t *testing.T) {
	var scan scanner
	item, rest, err := nextValue(jsonBig, &scan)
	if err != nil {
		t.Fatalf("nextValue: %s", err)
	}
	if len(item) != len(jsonBig) || &item[0] != &jsonBig[0] {
		t.Errorf("invalid item: %d %d", len(item), len(jsonBig))
	}
	if len(rest) != 0 {
		t.Errorf("invalid rest: %d", len(rest))
	}

	item, rest, err = nextValue(append(jsonBig, []byte("HELLO WORLD")...), &scan)
	if err != nil {
		t.Fatalf("nextValue extra: %s", err)
	}
	if len(item) != len(jsonBig) {
		t.Errorf("invalid item: %d %d", len(item), len(jsonBig))
	}
	if string(rest) != "HELLO WORLD" {
		t.Errorf("invalid rest: %d", len(rest))
	}
}

func BenchmarkSkipValue(b *testing.B) {
	var scan scanner
	for i := 0; i < b.N; i++ {
		nextValue(jsonBig, &scan)
	}
	b.SetBytes(int64(len(jsonBig)))
}

func diff(t *testing.T, a, b []byte) {
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
	if len(b) > 20 {
		return b[0:20]
	}
	return b
}

// Generate a random JSON object.

var jsonBig []byte

func init() {
	b, err := Marshal(genValue(10000))
	if err != nil {
		panic(err)
	}
	jsonBig = b
}

func genValue(n int) interface{} {
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
	n := int(math.Fabs(rand.NormFloat64()*stddev + stddev/2))
	c := make([]int, n)
	for i := range c {
		f := math.Fabs(rand.NormFloat64()*64 + 32)
		if f > 0x10ffff {
			f = 0x10ffff
		}
		c[i] = int(f)
	}
	return string(c)
}

func genArray(n int) []interface{} {
	f := int(math.Fabs(rand.NormFloat64()) * math.Fmin(10, float64(n/2)))
	if f > n {
		f = n
	}
	x := make([]interface{}, int(f))
	for i := range x {
		x[i] = genValue(((i+1)*n)/f - (i*n)/f)
	}
	return x
}

func genMap(n int) map[string]interface{} {
	f := int(math.Fabs(rand.NormFloat64()) * math.Fmin(10, float64(n/2)))
	if f > n {
		f = n
	}
	if n > 0 && f == 0 {
		f = 1
	}
	x := make(map[string]interface{})
	for i := 0; i < f; i++ {
		x[genString(10)] = genValue(((i+1)*n)/f - (i*n)/f)
	}
	return x
}
