// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonwire

import (
	"bytes"
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

	"encoding/json/internal/jsonflags"
)

const scanFiller = "acdeghijklmopqsvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !#$%'()*+,-./:;=?@[]^_`{|}~\x7f"

var scanEncodeSpecials = []string{
	"\"", "\\", "\x00", "\x01", "\b", "\t", "\n", "\f", "\r", "\x1f", "<", ">", "&", "\x7f", " ",
	"\x80", "\xbf", "\xc0", "\xff", "\xc3", "\u00e9", "\u2027", "\u2028", "\u2029", "\ufffd",
	"\U0001f600", "\xf0\x9f\x98", "\xed\xa0\x80", "\xe2\x80",
}

var scanDecodeSpecials = []string{
	"\"", "\\", `\n`, `\"`, `\\`, `\/`, `\u0041`, `\x`, "\x00", "\x01", "\t", "\x1f", "<", "\x7f",
	"\x80", "\xff", "\xc3", "\u00e9", "\u2028", "\ufffd", "\U0001f600", "\xf0\x9f\x98", "\xe2\x80",
}

var scanByteLengths = []int{1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100}

var scanByteSpecials = func() []string {
	specials := make([]string, 256)
	for i := range specials {
		specials[i] = string([]byte{byte(i)})
	}
	return specials
}()

var scanBuf [16 + 256 + 64]byte

type scanCase struct {
	n, pos             int
	sp, prefix, suffix string
}

func (c scanCase) String() string {
	return fmt.Sprintf("n=%d pos=%d special=%q prefix=%q suffix=%q", c.n, c.pos, c.sp, c.prefix, c.suffix)
}

func (c scanCase) input() []byte {
	buf := scanBuf[:]
	for i := range buf {
		buf[i] = '"'
	}
	size := len(c.prefix) + c.n + len(c.suffix)
	off := (c.n + c.pos) % 16
	in := buf[off : off+size]
	body := in[len(c.prefix) : len(c.prefix)+c.n]
	copy(in, c.prefix)
	for i := range body {
		body[i] = scanFiller[(i*7+c.pos)%len(scanFiller)]
	}
	copy(body[c.pos:], c.sp)
	copy(in[len(c.prefix)+c.n:], c.suffix)
	return in
}

type scanSweep struct {
	lengths  []int
	specials []string
	shape    scanCase
}

func (s scanSweep) run(check func(in []byte, c scanCase)) {
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

func scanLengths(limit int) []int {
	if testing.Short() {
		limit = min(limit, 64)
	}
	lengths := make([]int, limit+1)
	for i := range lengths {
		lengths[i] = i
	}
	return lengths
}

func runScanSweeps(shape scanCase, specials []string, limit int, check func(in []byte, c scanCase)) {
	scanSweep{lengths: scanLengths(limit), specials: specials, shape: shape}.run(check)
	scanSweep{lengths: scanByteLengths, specials: scanByteSpecials, shape: shape}.run(check)
}

var scanShortEscapes = map[rune]string{'\b': `\b`, '\f': `\f`, '\n': `\n`, '\r': `\r`, '\t': `\t`}

var scanUnescapes = map[byte]string{'"': `"`, '\\': `\`, '/': `/`, 'b': "\b", 'f': "\f", 'n': "\n", 'r': "\r", 't': "\t"}

const scanHex = "0123456789abcdef"

func refAppendEscapedUnit(out []byte, u uint16) []byte {
	return append(out, '\\', 'u', scanHex[u>>12&0xf], scanHex[u>>8&0xf], scanHex[u>>4&0xf], scanHex[u&0xf])
}

func refMustEscapeUnit(r rune, flags jsonflags.Flags) bool {
	switch r {
	case '<', '>', '&':
		return flags.Get(jsonflags.EscapeForHTML)
	case '\u2028', '\u2029':
		return flags.Get(jsonflags.EscapeForJS)
	}
	return r < ' '
}

func refAppendRune(out []byte, r rune, raw []byte, flags jsonflags.Flags) []byte {
	switch short := scanShortEscapes[r]; {
	case r == utf8.RuneError && len(raw) == 1:
		return append(out, "\ufffd"...)
	case r == '"' || r == '\\':
		return append(out, '\\', byte(r))
	case short != "":
		return append(out, short...)
	case refMustEscapeUnit(r, flags):
		return refAppendEscapedUnit(out, uint16(r))
	}
	return append(out, raw...)
}

func refAppendQuote(src []byte, flags jsonflags.Flags) ([]byte, error) {
	out := []byte{'"'}
	var invalid bool
	for i := 0; i < len(src); {
		r, size := utf8.DecodeRune(src[i:])
		invalid = invalid || (r == utf8.RuneError && size == 1)
		out = refAppendRune(out, r, src[i:i+size], flags)
		i += size
	}
	out = append(out, '"')
	if invalid && !flags.Get(jsonflags.AllowInvalidUTF8) {
		return out, ErrInvalidUTF8
	}
	return out, nil
}

func refNeedEscape(src []byte) bool {
	for i := 0; i < len(src); {
		r, size := utf8.DecodeRune(src[i:])
		switch {
		case r == utf8.RuneError, r == '\u2028', r == '\u2029':
			return true
		case r < ' ', r == '"', r == '\\', r == '<', r == '>', r == '&':
			return true
		}
		i += size
	}
	return false
}

func refConsumeString(b []byte, validateUTF8 bool) (n int, flags ValueFlags, err error) {
	if len(b) == 0 {
		return 0, 0, io.ErrUnexpectedEOF
	}
	if b[0] != '"' {
		return 0, 0, NewInvalidCharacterError(b, `at start of string (expecting '"')`)
	}
	n = 1
	for n < len(b) {
		var done bool
		if n, done, err = refConsumeStep(b, n, &flags, validateUTF8); done {
			return n, flags, err
		}
	}
	return n, flags, io.ErrUnexpectedEOF
}

func refConsumeStep(b []byte, n int, flags *ValueFlags, validateUTF8 bool) (int, bool, error) {
	switch c := b[n]; {
	case c == '"':
		return n + 1, true, nil
	case c < ' ':
		flags.Join(stringNonVerbatim | stringNonCanonical)
		return n, true, NewInvalidCharacterError(b[n:], "in string (expecting non-control character)")
	case c == '\\':
		return refConsumeEscape(b, n, flags)
	case c < utf8.RuneSelf:
		return n + 1, false, nil
	}
	return refConsumeHigh(b, n, flags, validateUTF8)
}

func refConsumeEscape(b []byte, n int, flags *ValueFlags) (int, bool, error) {
	flags.Join(stringNonVerbatim)
	if len(b) < n+2 {
		return n, true, io.ErrUnexpectedEOF
	}
	switch b[n+1] {
	case '"', '\\', 'b', 'f', 'n', 'r', 't':
		return n + 2, false, nil
	case '/':
		flags.Join(stringNonCanonical)
		return n + 2, false, nil
	case 'u':
		flags.Join(stringNonCanonical)
		return n + 6, false, nil
	}
	flags.Join(stringNonCanonical)
	return n, true, NewInvalidEscapeSequenceError(b[n : n+2])
}

func refConsumeHigh(b []byte, n int, flags *ValueFlags, validateUTF8 bool) (int, bool, error) {
	if _, size := utf8.DecodeRune(b[n:]); size > 1 {
		return n + size, false, nil
	}
	if !utf8.FullRune(b[n:]) {
		return n, true, io.ErrUnexpectedEOF
	}
	flags.Join(stringNonVerbatim | stringNonCanonical)
	if validateUTF8 {
		return n, true, ErrInvalidUTF8
	}
	return n + 1, false, nil
}

func refUnescape(b []byte) (dec string, size int, ok bool) {
	if dec, ok := scanUnescapes[b[1]]; ok {
		return dec, 2, true
	}
	if b[1] == 'u' && len(b) >= 6 {
		if v, err := strconv.ParseUint(string(b[2:6]), 16, 16); err == nil {
			return string(rune(v)), 6, true
		}
	}
	return "", 0, false
}

type refUnquoter struct {
	src     []byte
	out     []byte
	n       int
	pending error
}

func refAppendUnquote(src []byte) ([]byte, error) {
	if len(src) == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	if src[0] != '"' {
		return nil, NewInvalidCharacterError(src, `at start of string (expecting '"')`)
	}
	u := refUnquoter{src: src, n: 1}
	for u.n < len(src) {
		if done, err := u.step(); done {
			return u.out, err
		}
	}
	return u.out, io.ErrUnexpectedEOF
}

func (u *refUnquoter) step() (done bool, err error) {
	switch c := u.src[u.n]; {
	case c == '"':
		return true, u.finish()
	case c < ' ':
		return true, NewInvalidCharacterError(u.src[u.n:], "in string (expecting non-control character)")
	case c == '\\':
		return u.escape()
	case c < utf8.RuneSelf:
		u.out = append(u.out, c)
		u.n++
		return false, nil
	}
	return u.high()
}

func (u *refUnquoter) finish() error {
	if u.n+1 < len(u.src) {
		return NewInvalidCharacterError(u.src[u.n+1:], "after string value")
	}
	return u.pending
}

func (u *refUnquoter) escape() (done bool, err error) {
	if len(u.src) < u.n+2 {
		return true, io.ErrUnexpectedEOF
	}
	dec, size, ok := refUnescape(u.src[u.n:])
	if !ok {
		return true, NewInvalidEscapeSequenceError(u.src[u.n : u.n+2])
	}
	u.out = append(u.out, dec...)
	u.n += size
	return false, nil
}

func (u *refUnquoter) high() (done bool, err error) {
	if _, size := utf8.DecodeRune(u.src[u.n:]); size > 1 {
		u.out = append(u.out, u.src[u.n:u.n+size]...)
		u.n += size
		return false, nil
	}
	if !utf8.FullRune(u.src[u.n:]) {
		return true, io.ErrUnexpectedEOF
	}
	u.out = append(u.out, "\ufffd"...)
	u.n++
	u.pending = ErrInvalidUTF8
	return false, nil
}

func refReformatString(src []byte, flags jsonflags.Flags) ([]byte, int, error) {
	n, valFlags, err := refConsumeString(src, !flags.Get(jsonflags.AllowInvalidUTF8))
	if err != nil {
		return nil, n, err
	}
	preserve := flags.Get(jsonflags.PreserveRawStrings)
	if !flags.Get(jsonflags.AnyEscape) && (valFlags.IsCanonical() || preserve) {
		return bytes.Clone(src[:n]), n, nil
	}
	if preserve {
		return refReescapeRaw(src[:n], flags), n, nil
	}
	unquoted, _ := refAppendUnquote(src[:n])
	out, _ := refAppendQuote(unquoted, flags)
	return out, n, nil
}

func refReescapeRaw(src []byte, flags jsonflags.Flags) []byte {
	var out []byte
	for i := 0; i < len(src); {
		r, size := utf8.DecodeRune(src[i:])
		if refMustEscapeUnit(r, flags) {
			out = refAppendEscapedUnit(out, uint16(r))
		} else {
			out = append(out, src[i:i+size]...)
		}
		i += size
	}
	return out
}

func makeScanFlags(set jsonflags.Bools) jsonflags.Flags {
	var flags jsonflags.Flags
	flags.Set(set | 1)
	return flags
}

func checkScanAppendQuote(t *testing.T, in []byte, c scanCase) {
	t.Helper()
	for _, set := range []jsonflags.Bools{
		0,
		jsonflags.EscapeForHTML,
		jsonflags.EscapeForJS,
		jsonflags.EscapeForHTML | jsonflags.EscapeForJS | jsonflags.AllowInvalidUTF8,
	} {
		flags := makeScanFlags(set)
		got, gotErr := AppendQuote(nil, in, &flags)
		want, wantErr := refAppendQuote(in, flags)
		if !bytes.Equal(got, want) || !errors.Is(gotErr, wantErr) {
			t.Fatalf("AppendQuote(%q, flags=%v) [%v] = (%q, %v), want (%q, %v)", in, set, c, got, gotErr, want, wantErr)
		}
	}
}

func checkScanNeedEscape(t *testing.T, in []byte, c scanCase) {
	t.Helper()
	if got, want := NeedEscape(in), refNeedEscape(in); got != want {
		t.Fatalf("NeedEscape(%q) [%v] = %v, want %v", in, c, got, want)
	}
}

func checkScanConsume(t *testing.T, in []byte, c scanCase) {
	t.Helper()
	for _, validate := range []bool{false, true} {
		var gotFlags ValueFlags
		gotN, gotErr := ConsumeString(&gotFlags, in, validate)
		wantN, wantFlags, wantErr := refConsumeString(in, validate)
		if gotN != wantN || gotFlags != wantFlags || !reflect.DeepEqual(gotErr, wantErr) {
			t.Fatalf("ConsumeString(%q, %v) [%v] = (%d, %v, %v), want (%d, %v, %v)",
				in, validate, c, gotN, gotFlags, gotErr, wantN, wantFlags, wantErr)
		}
	}
}

func checkScanResume(t *testing.T, in []byte, c scanCase) {
	t.Helper()
	wantN, wantFlags, wantErr := refConsumeString(in, false)
	for cut := 1; cut <= len(in); cut++ {
		var flags ValueFlags
		n, err := ConsumeStringStreaming(&flags, in[:cut], 0, false)
		if errors.Is(err, io.ErrUnexpectedEOF) {
			n, err = ConsumeStringFinal(&flags, in, n, false)
		}
		if n != wantN || flags != wantFlags || !reflect.DeepEqual(err, wantErr) {
			t.Fatalf("ConsumeStringStreaming+ConsumeStringFinal(%q, cut=%d) [%v] = (%d, %v, %v), want (%d, %v, %v)",
				in, cut, c, n, flags, err, wantN, wantFlags, wantErr)
		}
	}
}

func checkScanUnquote(t *testing.T, in []byte, c scanCase) {
	t.Helper()
	got, gotErr := AppendUnquote(nil, in)
	want, wantErr := refAppendUnquote(in)
	if !bytes.Equal(got, want) || !reflect.DeepEqual(gotErr, wantErr) {
		t.Fatalf("AppendUnquote(%q) [%v] = (%q, %v), want (%q, %v)", in, c, got, gotErr, want, wantErr)
	}
}

func checkScanReformat(t *testing.T, in []byte, c scanCase) {
	t.Helper()
	for _, set := range []jsonflags.Bools{
		jsonflags.PreserveRawStrings | jsonflags.EscapeForHTML | jsonflags.AllowInvalidUTF8,
		jsonflags.PreserveRawStrings | jsonflags.EscapeForJS | jsonflags.AllowInvalidUTF8,
		jsonflags.PreserveRawStrings | jsonflags.EscapeForHTML | jsonflags.EscapeForJS,
		jsonflags.EscapeForHTML | jsonflags.EscapeForJS | jsonflags.AllowInvalidUTF8,
		jsonflags.AllowInvalidUTF8,
	} {
		flags := makeScanFlags(set)
		got, gotN, gotErr := ReformatString(nil, in, &flags)
		want, wantN, wantErr := refReformatString(in, flags)
		if !bytes.Equal(got, want) || gotN != wantN || !reflect.DeepEqual(gotErr, wantErr) {
			t.Fatalf("ReformatString(%q, flags=%v) [%v] = (%q, %d, %v), want (%q, %d, %v)",
				in, set, c, got, gotN, gotErr, want, wantN, wantErr)
		}
	}
}

func TestScanAppendQuote(t *testing.T) {
	runScanSweeps(scanCase{}, scanEncodeSpecials, 200, func(in []byte, c scanCase) {
		checkScanAppendQuote(t, in, c)
	})
}

func TestScanNeedEscape(t *testing.T) {
	runScanSweeps(scanCase{}, scanEncodeSpecials, 200, func(in []byte, c scanCase) {
		checkScanNeedEscape(t, in, c)
	})
}

func TestScanConsumeString(t *testing.T) {
	for _, suffix := range []string{"", `"`} {
		runScanSweeps(scanCase{prefix: `"`, suffix: suffix}, scanDecodeSpecials, 200, func(in []byte, c scanCase) {
			checkScanConsume(t, in, c)
		})
	}
}

func TestScanConsumeStringResume(t *testing.T) {
	sweep := scanSweep{
		lengths:  []int{16, 17, 33, 64, 65},
		specials: scanDecodeSpecials,
		shape:    scanCase{prefix: `"`, suffix: `"`},
	}
	sweep.run(func(in []byte, c scanCase) {
		checkScanResume(t, in, c)
	})
}

func TestScanAppendUnquote(t *testing.T) {
	for _, suffix := range []string{"", `"`, `"x`} {
		runScanSweeps(scanCase{prefix: `"`, suffix: suffix}, scanDecodeSpecials, 200, func(in []byte, c scanCase) {
			checkScanUnquote(t, in, c)
		})
	}
}

var scanRunLengths = []int{0, 1, 5, 7, 8, 15, 16, 23, 24, 31, 32, 33, 55, 56, 57, 63, 64, 100}

var scanBenignSpecials = []string{
	`\n`, `\"`, `\\`, `\/`, `\u0041`, "<", "\x7f", "\u00e9", "\u2028", "\ufffd", "\U0001f600", "\xff", "\xc3",
}

func scanPatternBody(rng *rand.Rand, specials []string) []byte {
	var body []byte
	for range 1 + rng.IntN(8) {
		for range scanRunLengths[rng.IntN(len(scanRunLengths))] {
			body = append(body, scanFiller[rng.IntN(len(scanFiller))])
		}
		body = append(body, specials[rng.IntN(len(specials))]...)
	}
	for range scanRunLengths[rng.IntN(len(scanRunLengths))] {
		body = append(body, scanFiller[rng.IntN(len(scanFiller))])
	}
	return body
}

func scanQuoted(body []byte, suffix string) []byte {
	return append(append([]byte{'"'}, body...), suffix...)
}

func scanPatternCount() int {
	if testing.Short() {
		return 500
	}
	return 4000
}

func TestScanEncodePatterns(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for range scanPatternCount() {
		body := scanPatternBody(rng, scanEncodeSpecials)
		c := scanCase{n: len(body), sp: "pattern"}
		checkScanAppendQuote(t, body, c)
		checkScanNeedEscape(t, body, c)
	}
}

func TestScanDecodePatterns(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for i := range scanPatternCount() {
		specials := scanBenignSpecials
		if i%4 == 3 {
			specials = scanDecodeSpecials
		}
		body := scanPatternBody(rng, specials)
		for _, suffix := range []string{"", `"`, `"x`} {
			in := scanQuoted(body, suffix)
			c := scanCase{n: len(body), sp: "pattern", prefix: `"`, suffix: suffix}
			checkScanConsume(t, in, c)
			checkScanUnquote(t, in, c)
		}
		checkScanReformat(t, scanQuoted(body, `"`), scanCase{n: len(body), sp: "pattern", prefix: `"`, suffix: `"`})
	}
}

func TestScanReformatString(t *testing.T) {
	runScanSweeps(scanCase{prefix: `"`, suffix: `"`}, scanDecodeSpecials, 64, func(in []byte, c scanCase) {
		checkScanReformat(t, in, c)
	})
}

type scanTail struct {
	n, p1, back int
	s1, s2      string
}

func (t scanTail) body() ([]byte, bool) {
	p2 := t.n - len(t.s2) - t.back
	if p2 < t.p1+len(t.s1) {
		return nil, false
	}
	body := make([]byte, t.n)
	for i := range body {
		body[i] = scanFiller[(i*7+t.p1)%len(scanFiller)]
	}
	copy(body[t.p1:], t.s1)
	copy(body[p2:], t.s2)
	return body, true
}

var scanTailShapes = [][2]int{{16, 0}, {16, 1}, {16, 2}, {24, 0}, {24, 1}, {24, 2}, {32, 0}, {32, 1}, {32, 2}}

func scanTailLengths() []int {
	if testing.Short() {
		return []int{24, 31, 32, 33, 40, 48}
	}
	lengths := make([]int, 0, 25)
	for n := 24; n <= 48; n++ {
		lengths = append(lengths, n)
	}
	return lengths
}

func scanSpecialPairs(specials []string) [][2]string {
	pairs := make([][2]string, 0, len(specials)*len(specials))
	for _, a := range specials {
		for _, b := range specials {
			pairs = append(pairs, [2]string{a, b})
		}
	}
	return pairs
}

func scanTailBodies(specials []string, emit func(body []byte)) {
	pairs := scanSpecialPairs(specials)
	for _, n := range scanTailLengths() {
		for _, shape := range scanTailShapes {
			for _, pair := range pairs {
				tail := scanTail{n: n, p1: shape[0], back: shape[1], s1: pair[0], s2: pair[1]}
				if body, ok := tail.body(); ok {
					emit(body)
				}
			}
		}
	}
}

func TestScanTailSpecials(t *testing.T) {
	scanTailBodies(scanEncodeSpecials, func(body []byte) {
		c := scanCase{n: len(body), sp: "tail"}
		checkScanAppendQuote(t, body, c)
		checkScanNeedEscape(t, body, c)
	})
	scanTailBodies(scanDecodeSpecials, func(body []byte) {
		for _, suffix := range []string{"", `"`} {
			c := scanCase{n: len(body), sp: "tail", prefix: `"`, suffix: suffix}
			in := scanQuoted(body, suffix)
			checkScanConsume(t, in, c)
			checkScanUnquote(t, in, c)
		}
	})
}

var scanBenchKinds = []string{"clean", "sparse", "dense", "mixed"}

func skipScanAllocTest(t *testing.T) {
	t.Helper()
	testenv.SkipIfOptimizationOff(t)
	if race.Enabled {
		t.Skip("allocation counts differ under the race detector")
	}
}

func TestScanEncodeAllocs(t *testing.T) {
	skipScanAllocTest(t)
	for _, kind := range scanBenchKinds {
		src := scanBenchString(kind, 4096)
		str := string(src)
		dst := make([]byte, 0, 2*len(src)+2)
		flags := makeScanFlags(jsonflags.EscapeForHTML)
		got := map[string]float64{
			"AppendQuote([]byte)": testing.AllocsPerRun(20, func() { dst, _ = AppendQuote(dst[:0], src, &flags) }),
			"AppendQuote(string)": testing.AllocsPerRun(20, func() { dst, _ = AppendQuote(dst[:0], []byte(str), &flags) }),
			"NeedEscape([]byte)":  testing.AllocsPerRun(20, func() { _ = NeedEscape(src) }),
			"NeedEscape(string)":  testing.AllocsPerRun(20, func() { _ = NeedEscape([]byte(str)) }),
		}
		for name, allocs := range got {
			if allocs != 0 {
				t.Errorf("%s allocs per run for %s/4096 = %v, want 0", name, kind, allocs)
			}
		}
	}
}

func TestScanDecodeAllocs(t *testing.T) {
	skipScanAllocTest(t)
	for _, kind := range scanBenchKinds {
		var flags jsonflags.Flags
		quoted, _ := AppendQuote(nil, scanBenchString(kind, 4096), &flags)
		dst := make([]byte, 0, len(quoted))
		got := map[string]float64{
			"ConsumeString([]byte)": testing.AllocsPerRun(20, func() {
				var valFlags ValueFlags
				_, _ = ConsumeString(&valFlags, quoted, true)
			}),
			"AppendUnquote([]byte)": testing.AllocsPerRun(20, func() { dst, _ = AppendUnquote(dst[:0], quoted) }),
		}
		for name, allocs := range got {
			if allocs != 0 {
				t.Errorf("%s allocs per run for %s/4096 = %v, want 0", name, kind, allocs)
			}
		}
	}
}

func scanBenchString(kind string, size int) []byte {
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
	return b
}

func benchScanStrings(b *testing.B, run func(b *testing.B, src []byte)) {
	for _, kind := range scanBenchKinds {
		for _, size := range []int{32, 256, 4096} {
			b.Run(fmt.Sprintf("%s/%d", kind, size), func(b *testing.B) {
				run(b, scanBenchString(kind, size))
			})
		}
	}
}

func BenchmarkScanAppendQuote(b *testing.B) {
	benchScanStrings(b, func(b *testing.B, src []byte) {
		var flags jsonflags.Flags
		dst := make([]byte, 0, 2*len(src)+2)
		b.SetBytes(int64(len(src)))
		b.ReportAllocs()
		for range b.N {
			dst, _ = AppendQuote(dst[:0], src, &flags)
		}
	})
}

func BenchmarkScanConsumeString(b *testing.B) {
	benchScanStrings(b, func(b *testing.B, src []byte) {
		var flags jsonflags.Flags
		quoted, _ := AppendQuote(nil, src, &flags)
		b.SetBytes(int64(len(quoted)))
		b.ReportAllocs()
		for range b.N {
			var valFlags ValueFlags
			if n, err := ConsumeString(&valFlags, quoted, true); n != len(quoted) || err != nil {
				b.Fatalf("ConsumeString = (%d, %v), want (%d, nil)", n, err, len(quoted))
			}
		}
	})
}

func BenchmarkScanAppendUnquote(b *testing.B) {
	benchScanStrings(b, func(b *testing.B, src []byte) {
		var flags jsonflags.Flags
		quoted, _ := AppendQuote(nil, src, &flags)
		dst := make([]byte, 0, len(quoted))
		b.SetBytes(int64(len(quoted)))
		b.ReportAllocs()
		for range b.N {
			var err error
			if dst, err = AppendUnquote(dst[:0], quoted); err != nil {
				b.Fatalf("AppendUnquote error: %v", err)
			}
		}
	})
}
