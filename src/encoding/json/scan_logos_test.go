// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"internal/race"
	"internal/testenv"
	"io"
	"math/rand/v2"
	"reflect"
	"strconv"
	"testing"
	"unicode/utf8"
)

const logosFiller = "acdeghijklmopqsvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !#$%'()*+,-./:;=?@[]^_`{|}~\x7f"

var logosEncodeSpecials = []string{
	"\"", "\\", "\x00", "\x01", "\b", "\t", "\n", "\f", "\r", "\x1f", "<", ">", "&", "\x7f", " ",
	"\x80", "\xbf", "\xc0", "\xff", "\xc3", "\u00e9", "\u2027", "\u2028", "\u2029", "\ufffd",
	"\U0001f600", "\xf0\x9f\x98", "\xed\xa0\x80", "\xe2\x80",
}

var logosDecodeSpecials = []string{
	"\"", "\\", `\n`, `\"`, `\\`, `\/`, `\u0041`, `\x`, "\x00", "\x01", "\t", "\x1f", "<", "\x7f",
	"\x80", "\xff", "\xc3", "\u00e9", "\u2028", "\ufffd", "\U0001f600", "\xf0\x9f\x98", "\xe2\x80",
}

var logosByteLengths = []int{1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100}

var logosByteSpecials = func() []string {
	specials := make([]string, 256)
	for i := range specials {
		specials[i] = string([]byte{byte(i)})
	}
	return specials
}()

var logosBuf [16 + 256 + 64]byte

type logosCase struct {
	n, pos             int
	sp, prefix, suffix string
}

func (c logosCase) String() string {
	return fmt.Sprintf("n=%d pos=%d special=%q prefix=%q suffix=%q", c.n, c.pos, c.sp, c.prefix, c.suffix)
}

func (c logosCase) input() []byte {
	buf := logosBuf[:]
	for i := range buf {
		buf[i] = '"'
	}
	size := len(c.prefix) + c.n + len(c.suffix)
	off := (c.n + c.pos) % 16
	in := buf[off : off+size]
	body := in[len(c.prefix) : len(c.prefix)+c.n]
	copy(in, c.prefix)
	for i := range body {
		body[i] = logosFiller[(i*7+c.pos)%len(logosFiller)]
	}
	copy(body[c.pos:], c.sp)
	copy(in[len(c.prefix)+c.n:], c.suffix)
	return in
}

type logosSweep struct {
	lengths  []int
	specials []string
	shape    logosCase
}

func (s logosSweep) run(check func(in []byte, c logosCase)) {
	specials := append([]string{""}, s.specials...)
	for _, n := range s.lengths {
		c := s.shape
		c.n = n
		for _, sp := range specials {
			c.sp = sp
			for c.pos = 0; c.pos+len(sp) <= n; c.pos++ {
				check(c.input(), c)
			}
		}
	}
}

func logosLengths(limit int) []int {
	if testing.Short() {
		limit = min(limit, 64)
	}
	lengths := make([]int, limit+1)
	for i := range lengths {
		lengths[i] = i
	}
	return lengths
}

func runLogosSweeps(shape logosCase, specials []string, limit int, check func(in []byte, c logosCase)) {
	logosSweep{lengths: logosLengths(limit), specials: specials, shape: shape}.run(check)
	logosSweep{lengths: logosByteLengths, specials: logosByteSpecials, shape: shape}.run(check)
}

var logosShortEscapes = map[rune]string{'\b': `\b`, '\f': `\f`, '\n': `\n`, '\r': `\r`, '\t': `\t`}

var logosUnescapes = map[byte]string{'"': `"`, '\\': `\`, '/': `/`, 'b': "\b", 'f': "\f", 'n': "\n", 'r': "\r", 't': "\t"}

const logosHex = "0123456789abcdef"

type logosQuoter struct {
	escapeHTML bool
	invalid    string
}

func logosAppendUnit(out []byte, u rune) []byte {
	return append(out, '\\', 'u', logosHex[u>>12&0xf], logosHex[u>>8&0xf], logosHex[u>>4&0xf], logosHex[u&0xf])
}

func (q logosQuoter) mustEscape(r rune) bool {
	switch r {
	case '<', '>', '&':
		return q.escapeHTML
	case '\u2028', '\u2029':
		return true
	}
	return r < ' '
}

func (q logosQuoter) appendRune(out []byte, r rune, raw string) []byte {
	switch short := logosShortEscapes[r]; {
	case r == utf8.RuneError && len(raw) == 1:
		return append(out, q.invalid...)
	case r == '"' || r == '\\':
		return append(out, '\\', byte(r))
	case short != "":
		return append(out, short...)
	case q.mustEscape(r):
		return logosAppendUnit(out, r)
	}
	return append(out, raw...)
}

func (q logosQuoter) quote(s string) string {
	out := []byte{'"'}
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		out = q.appendRune(out, r, s[i:i+size])
		i += size
	}
	return string(append(out, '"'))
}

func logosInvalidRendering(t *testing.T) string {
	t.Helper()
	got, err := json.Marshal("\xff")
	if err != nil {
		t.Fatalf("Marshal(%q) error: %v", "\xff", err)
	}
	switch string(got) {
	case `"\ufffd"`, "\"\ufffd\"":
		return string(got[1 : len(got)-1])
	}
	t.Fatalf("Marshal(%q) = %q, want a quoted replacement character", "\xff", got)
	return ""
}

func logosUnescape(b []byte) (dec string, size int, ok bool) {
	if dec, ok := logosUnescapes[b[1]]; ok {
		return dec, 2, true
	}
	if b[1] == 'u' && len(b) >= 6 {
		if v, err := strconv.ParseUint(string(b[2:6]), 16, 16); err == nil {
			return string(rune(v)), 6, true
		}
	}
	return "", 0, false
}

func logosDecodeQuoted(text []byte) (string, bool) {
	var out []byte
	for i := 1; i < len(text); {
		switch c := text[i]; {
		case c == '"':
			return string(out), i == len(text)-1
		case c < ' ':
			return "", false
		case c == '\\':
			dec, size, ok := logosUnescape(text[i:])
			if !ok {
				return "", false
			}
			out = append(out, dec...)
			i += size
		default:
			r, size := utf8.DecodeRune(text[i:])
			out = utf8.AppendRune(out, r)
			i += size
		}
	}
	return "", false
}

func checkLogosMarshal(t *testing.T, in []byte, c logosCase, invalid string) {
	t.Helper()
	s := string(in)
	got, err := json.Marshal(s)
	if want := (logosQuoter{escapeHTML: true, invalid: invalid}).quote(s); err != nil || string(got) != want {
		t.Fatalf("Marshal(%q) [%v] = (%q, %v), want (%q, nil)", s, c, got, err, want)
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s); err != nil {
		t.Fatalf("Encode(%q) [%v] error: %v", s, c, err)
	}
	if want := (logosQuoter{invalid: invalid}).quote(s) + "\n"; buf.String() != want {
		t.Fatalf("Encode(%q) with SetEscapeHTML(false) [%v] = %q, want %q", s, c, buf.Bytes(), want)
	}
}

func logosControlOffsets(c logosCase) []int64 {
	if len(c.sp) != 1 || c.sp[0] >= ' ' {
		return nil
	}
	index := len(c.prefix) + c.pos
	return []int64{int64(index), int64(index) + 1}
}

func logosOffsetAccepted(err error, c logosCase) bool {
	offsets := logosControlOffsets(c)
	if offsets == nil {
		return true
	}
	syntax, ok := errors.AsType[*json.SyntaxError](err)
	return ok && (syntax.Offset == offsets[0] || syntax.Offset == offsets[1])
}

func checkLogosUnmarshal(t *testing.T, in []byte, c logosCase) {
	t.Helper()
	want, ok := logosDecodeQuoted(in)
	var got string
	err := json.Unmarshal(in, &got)
	if (err == nil) != ok || (ok && got != want) {
		t.Fatalf("Unmarshal(%q) [%v] = (%q, %v), want (%q, ok=%v)", in, c, got, err, want, ok)
	}
	if !ok && !logosOffsetAccepted(err, c) {
		t.Fatalf("Unmarshal(%q) [%v] error %v has an unexpected offset", in, c, err)
	}
	if utf8.Valid(in) && json.Valid(in) != ok {
		t.Fatalf("Valid(%q) [%v] = %v, want %v", in, c, !ok, ok)
	}
}

func checkLogosUnmarshalNested(t *testing.T, in []byte, c logosCase) {
	t.Helper()
	want, ok := logosDecodeQuoted(in)

	var list any
	array := append(append([]byte(`["x",`), in...), ']')
	err := json.Unmarshal(array, &list)
	if (err == nil) != ok || (ok && !reflect.DeepEqual(list, []any{"x", want})) {
		t.Fatalf("Unmarshal(%q) [%v] = (%v, %v), want (%q, ok=%v)", array, c, list, err, want, ok)
	}

	var skipped struct {
		B int `json:"b"`
	}
	object := append(append([]byte(`{"a":`), in...), `,"b":1}`...)
	err = json.Unmarshal(object, &skipped)
	if (err == nil) != ok || (ok && skipped.B != 1) {
		t.Fatalf("Unmarshal(%q) [%v] = (%+v, %v), want (B=1, ok=%v)", object, c, skipped, err, ok)
	}
}

func TestLogosMarshalString(t *testing.T) {
	invalid := logosInvalidRendering(t)
	runLogosSweeps(logosCase{}, logosEncodeSpecials, 200, func(in []byte, c logosCase) {
		checkLogosMarshal(t, in, c, invalid)
	})
}

func TestLogosUnmarshalString(t *testing.T) {
	runLogosSweeps(logosCase{prefix: `"`, suffix: `"`}, logosDecodeSpecials, 200, func(in []byte, c logosCase) {
		checkLogosUnmarshal(t, in, c)
	})
}

var logosRunLengths = []int{0, 1, 5, 7, 8, 15, 16, 23, 24, 31, 32, 33, 55, 56, 57, 63, 64, 100}

var logosBenignSpecials = []string{
	`\n`, `\"`, `\\`, `\/`, `\u0041`, "<", "\x7f", "\u00e9", "\u2028", "\ufffd", "\U0001f600", "\xff", "\xc3",
}

func logosPatternBody(rng *rand.Rand, specials []string) []byte {
	var body []byte
	for range 1 + rng.IntN(8) {
		for range logosRunLengths[rng.IntN(len(logosRunLengths))] {
			body = append(body, logosFiller[rng.IntN(len(logosFiller))])
		}
		body = append(body, specials[rng.IntN(len(specials))]...)
	}
	for range logosRunLengths[rng.IntN(len(logosRunLengths))] {
		body = append(body, logosFiller[rng.IntN(len(logosFiller))])
	}
	return body
}

func logosPatternCount() int {
	if testing.Short() {
		return 500
	}
	return 4000
}

func TestLogosMarshalPatterns(t *testing.T) {
	invalid := logosInvalidRendering(t)
	rng := rand.New(rand.NewPCG(1, 2))
	for range logosPatternCount() {
		body := logosPatternBody(rng, logosEncodeSpecials)
		checkLogosMarshal(t, body, logosCase{n: len(body), sp: "pattern"}, invalid)
	}
}

func TestLogosUnmarshalPatterns(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for i := range logosPatternCount() {
		specials := logosBenignSpecials
		if i%4 == 3 {
			specials = logosDecodeSpecials
		}
		body := logosPatternBody(rng, specials)
		in := append(append([]byte{'"'}, body...), '"')
		c := logosCase{n: len(body), sp: "pattern", prefix: `"`, suffix: `"`}
		checkLogosUnmarshal(t, in, c)
		checkLogosUnmarshalNested(t, in, c)
	}
}

func TestLogosUnmarshalNestedString(t *testing.T) {
	sweep := logosSweep{
		lengths:  logosLengths(80),
		specials: logosDecodeSpecials,
		shape:    logosCase{prefix: `"`, suffix: `"`},
	}
	sweep.run(func(in []byte, c logosCase) {
		checkLogosUnmarshalNested(t, in, c)
	})
}

type logosTail struct {
	n, p1, back int
	s1, s2      string
}

func (t logosTail) body() ([]byte, bool) {
	p2 := t.n - len(t.s2) - t.back
	if p2 < t.p1+len(t.s1) {
		return nil, false
	}
	body := make([]byte, t.n)
	for i := range body {
		body[i] = logosFiller[(i*7+t.p1)%len(logosFiller)]
	}
	copy(body[t.p1:], t.s1)
	copy(body[p2:], t.s2)
	return body, true
}

var logosTailShapes = [][2]int{{16, 0}, {16, 1}, {16, 2}, {24, 0}, {24, 1}, {24, 2}, {32, 0}, {32, 1}, {32, 2}}

func logosTailLengths() []int {
	if testing.Short() {
		return []int{24, 31, 32, 33, 40, 48}
	}
	lengths := make([]int, 0, 25)
	for n := 24; n <= 48; n++ {
		lengths = append(lengths, n)
	}
	return lengths
}

func logosSpecialPairs(specials []string) [][2]string {
	pairs := make([][2]string, 0, len(specials)*len(specials))
	for _, a := range specials {
		for _, b := range specials {
			pairs = append(pairs, [2]string{a, b})
		}
	}
	return pairs
}

func logosTailBodies(specials []string, emit func(body []byte)) {
	pairs := logosSpecialPairs(specials)
	for _, n := range logosTailLengths() {
		for _, shape := range logosTailShapes {
			for _, pair := range pairs {
				tail := logosTail{n: n, p1: shape[0], back: shape[1], s1: pair[0], s2: pair[1]}
				if body, ok := tail.body(); ok {
					emit(body)
				}
			}
		}
	}
}

func TestLogosTailSpecials(t *testing.T) {
	invalid := logosInvalidRendering(t)
	logosTailBodies(logosEncodeSpecials, func(body []byte) {
		checkLogosMarshal(t, body, logosCase{n: len(body), sp: "tail"}, invalid)
	})
	logosTailBodies(logosDecodeSpecials, func(body []byte) {
		in := append(append([]byte{'"'}, body...), '"')
		c := logosCase{n: len(body), sp: "tail", prefix: `"`, suffix: `"`}
		checkLogosUnmarshal(t, in, c)
		checkLogosUnmarshalNested(t, in, c)
	})
}

func TestLogosMarshalAllocsDoNotGrowWithLength(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	if race.Enabled {
		t.Skip("allocation counts differ under the race detector")
	}
	for _, kind := range []string{"clean", "sparse", "dense", "mixed"} {
		short, long := logosBenchString(kind, 24), logosBenchString(kind, 4096)
		shortAllocs := testing.AllocsPerRun(50, func() { _, _ = json.Marshal(short) })
		longAllocs := testing.AllocsPerRun(50, func() { _, _ = json.Marshal(long) })
		if longAllocs > shortAllocs {
			t.Errorf("Marshal(%s) allocs per run: 4096-byte string %v > 24-byte string %v", kind, longAllocs, shortAllocs)
		}
	}
}

func logosBenchString(kind string, size int) string {
	units := []string{"\u00e9", "\u65e5", "\U0001f600"}
	sparse := "\"\\\n\t<"
	b := make([]byte, 0, size)
	for len(b) < size {
		i := len(b)
		switch {
		case kind == "dense" && i%8 == 7:
			b = append(b, sparse[(i/8)%len(sparse)])
		case kind == "sparse" && i%64 == 63:
			b = append(b, sparse[(i/64)%len(sparse)])
		case kind == "mixed" && i%24 == 20 && i+4 <= size:
			b = append(b, units[(i/24)%len(units)]...)
		default:
			b = append(b, 'a'+byte(i%26))
		}
	}
	return string(b)
}

func benchLogosStrings(b *testing.B, run func(b *testing.B, s string)) {
	for _, kind := range []string{"clean", "sparse", "dense", "mixed"} {
		for _, size := range []int{32, 256, 4096} {
			b.Run(fmt.Sprintf("%s/%d", kind, size), func(b *testing.B) {
				run(b, logosBenchString(kind, size))
			})
		}
	}
}

func BenchmarkLogosMarshalString(b *testing.B) {
	benchLogosStrings(b, func(b *testing.B, s string) {
		b.SetBytes(int64(len(s)))
		b.ReportAllocs()
		for range b.N {
			if _, err := json.Marshal(s); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkLogosEncodeStringNoHTML(b *testing.B) {
	benchLogosStrings(b, func(b *testing.B, s string) {
		enc := json.NewEncoder(io.Discard)
		enc.SetEscapeHTML(false)
		b.SetBytes(int64(len(s)))
		b.ReportAllocs()
		for range b.N {
			if err := enc.Encode(s); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkLogosUnmarshalString(b *testing.B) {
	benchLogosStrings(b, func(b *testing.B, s string) {
		data, err := json.Marshal(s)
		if err != nil {
			b.Fatal(err)
		}
		var got string
		b.SetBytes(int64(len(data)))
		b.ReportAllocs()
		for range b.N {
			if err := json.Unmarshal(data, &got); err != nil {
				b.Fatal(err)
			}
		}
	})
}
