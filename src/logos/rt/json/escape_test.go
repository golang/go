// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	stdjson "encoding/json"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"
)

func escapeCorpus() []string {
	var c []string
	for b := range 0x80 {
		c = append(c, string([]byte{byte(b)}))
		c = append(c, "a"+string([]byte{byte(b)})+"z")
	}
	c = append(c,
		"", "plain ascii text", "<>&", "a<b>c&d", "\u2028", "\u2029", "x\u2028y\u2029z",
		"\"quoted\"", `back\slash`, "tab\there\nnl\rcr\bbs\fff", "\x7f", "\x00\x01\x1f",
		"\u0080", "\u00e9", "\u07ff", "\u0800", "\uffff", "\ufffd", "日本語", "😀", "a😀b",
		"\xc3", "\xc3(", "\x80", "a\x80b", "\xbf", "\xe2\x82", "\xe2\x28\xa1", "\xf0\x9f\x98",
		"\xf0\x9f", "\xc0\x80", "\xc1\xbf", "\xe0\x80\x80", "\xf0\x80\x80\x80", "\xed\xa0\x80",
		"\xed\xbf\xbf", "\xf4\x90\x80\x80", "\xff", "\xfe", "\xe2\x80", "\xe2\x80\xa8x",
		"\xff<\u2028\x00\"\\", "<\xff>",
	)
	for r := rune(0x80); r < 0x10000; r += 0x1d3 {
		if r >= 0xd800 && r < 0xe000 {
			continue
		}
		c = append(c, string(r), "q"+string(r)+"<")
	}
	c = append(c, strings.Repeat("abcdefgh", 700))
	c = append(c, strings.Repeat("abcdefgh<", 700))
	c = append(c, strings.Repeat("é\u2028\xff\"x", 900))
	c = append(c, strings.Repeat("0123456789", 500)+"\xff"+strings.Repeat("9876543210", 500))
	return c
}

type quotedString struct {
	S string `json:",string"`
}

func stdEscaped(t testing.TB, s string) string {
	t.Helper()
	b, err := stdjson.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal(%q): %v", s, err)
	}
	return string(b[1 : len(b)-1])
}

func stdQuoted(t testing.TB, s string) string {
	t.Helper()
	b, err := stdjson.Marshal(quotedString{s})
	if err != nil {
		t.Fatalf("Marshal(%q): %v", s, err)
	}
	const prefix = `{"S":"`
	const suffix = `"}`
	if !strings.HasPrefix(string(b), prefix) || !strings.HasSuffix(string(b), suffix) {
		t.Fatalf("unexpected quoted payload %q", b)
	}
	return string(b[len(prefix) : len(b)-len(suffix)])
}

func checkEscape(t testing.TB, s string) {
	t.Helper()
	want := stdEscaped(t, s)
	got := AppendEscaped(nil, s)
	if string(got) != want {
		t.Fatalf("AppendEscaped(%q) = %q, want %q", s, got, want)
	}
	if n := EscapedLen(s); n != len(got) {
		t.Fatalf("EscapedLen(%q) = %d, want %d", s, n, len(got))
	}
	framed := append(append([]byte{'"'}, AppendEscaped(nil, s)...), '"')
	std, _ := stdjson.Marshal(s)
	if string(framed) != string(std) {
		t.Fatalf("framed %q != Marshal %q", framed, std)
	}
	prefix := []byte("pre")
	if got := AppendEscaped(prefix, s); string(got) != "pre"+want {
		t.Fatalf("AppendEscaped with prefix(%q) = %q", s, got)
	}

	wantQ := stdQuoted(t, s)
	gotQ := AppendQuoted(nil, s)
	if string(gotQ) != wantQ {
		t.Fatalf("AppendQuoted(%q) = %q, want %q", s, gotQ, wantQ)
	}
	if n := QuotedLen(s); n != len(gotQ) {
		t.Fatalf("QuotedLen(%q) = %d, want %d", s, n, len(gotQ))
	}
	if got := AppendQuoted(prefix, s); string(got) != "pre"+wantQ {
		t.Fatalf("AppendQuoted with prefix(%q) = %q", s, got)
	}
}

func TestEscapeParity(t *testing.T) {
	for _, s := range escapeCorpus() {
		checkEscape(t, s)
	}
}

func TestEscapeInvalidUTF8Rule(t *testing.T) {
	got := string(AppendEscaped(nil, "a\xffb"))
	if got != "a"+invalidUTF8Escape+"b" {
		t.Fatalf("AppendEscaped invalid = %q, want rule text %q", got, invalidUTF8Escape)
	}
	if got != stdEscaped(t, "a\xffb") {
		t.Fatalf("invalid-UTF-8 rule disagrees with encoding/json: %q", got)
	}
}

func TestEscapeFastPathSingleCopy(t *testing.T) {
	s := strings.Repeat("no escapes here ", 64)
	buf := make([]byte, 0, len(s))
	allocs := testing.AllocsPerRun(100, func() {
		buf = AppendEscaped(buf[:0], s)
	})
	if allocs != 0 {
		t.Fatalf("fast path allocated %v times", allocs)
	}
	if string(buf) != s {
		t.Fatal("fast path altered the string")
	}
}

func TestEscapeLenAllocFree(t *testing.T) {
	s := "mixed <content> \u2028 with \xff bytes and \"quotes\""
	allocs := testing.AllocsPerRun(100, func() {
		_ = EscapedLen(s)
		_ = QuotedLen(s)
	})
	if allocs != 0 {
		t.Fatalf("len helpers allocated %v times", allocs)
	}
}

func FuzzEscape(f *testing.F) {
	for _, s := range escapeCorpus() {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		checkEscape(t, s)
	})
}

const diffMaxLen = 200

func diffPlainByte(j int) byte {
	const plain = " !#$%()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f"
	return plain[j%len(plain)]
}

func forEachPlacement(seqs []string, visit func(body []byte)) {
	filler := make([]byte, diffMaxLen)
	for j := range filler {
		filler[j] = diffPlainByte(j)
	}
	work := bytes.Clone(filler)
	for n := range diffMaxLen + 1 {
		visit(work[:n:n])
		for _, seq := range seqs {
			for pos := 0; pos+len(seq) <= n; pos++ {
				copy(work[pos:], seq)
				visit(work[:n:n])
				copy(work[pos:pos+len(seq)], filler[pos:])
			}
		}
	}
}

var escapeSequences = []string{
	"\x00", "\x08", "\t", "\n", "\x1f", " ", "!", "\"", "&", "<", ">", "\\", "~", "\x7f",
	"\x80", "\xbf", "\xc2", "\xc3", "\xe2", "\xf0", "\xf4", "\xff",
	"\u00e9", "\u07ff", "\u0800", "\u2027", "\u2028", "\u2029", "\u202a", "\ufffd", "\U0001f600",
	"\xe2\x80", "\xf0\x9f\x98", "\xc0\x80", "\xe0\x80\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80",
}

var lenSequences = []string{
	"\x00", "\n", "\"", "\\", "<", ">", "&", "\x7f", "\x80", "\xff",
	"\u00e9", "\u2028", "\U0001f600", "\xe2\x80",
}

func refFirstEscape(s string, from int) int {
	for off, r := range s[from:] {
		i := from + off
		if r < 0x20 || r == '"' || r == '\\' || r == '<' || r == '>' || r == '&' || r == '\u2028' || r == '\u2029' {
			return i
		}
		if r == utf8.RuneError && !strings.HasPrefix(s[i:], "\ufffd") {
			return i
		}
	}
	return len(s)
}

func refEscape(s string) string {
	var sb strings.Builder
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		unit := s[i : i+size]
		i += size
		switch r {
		case '"', '\\':
			sb.WriteByte('\\')
			sb.WriteString(unit)
		case '\b':
			sb.WriteString(`\b`)
		case '\f':
			sb.WriteString(`\f`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			sb.WriteString(`\r`)
		case '\t':
			sb.WriteString(`\t`)
		case '<', '>', '&', '\u2028', '\u2029':
			fmt.Fprintf(&sb, `\u%04x`, r)
		default:
			switch {
			case r < 0x20:
				fmt.Fprintf(&sb, `\u%04x`, r)
			case r == utf8.RuneError && size == 1:
				sb.WriteString(invalidUTF8Escape)
			default:
				sb.WriteString(unit)
			}
		}
	}
	return sb.String()
}

func refQuoted(s string) string {
	inner := `"` + refEscape(s) + `"`
	var sb strings.Builder
	for i := range len(inner) {
		if inner[i] == '"' || inner[i] == '\\' {
			sb.WriteByte('\\')
		}
		sb.WriteByte(inner[i])
	}
	return sb.String()
}

func TestFirstEscapeRules(t *testing.T) {
	cases := []struct {
		s    string
		from int
		want int
	}{
		{"", 0, 0},
		{"plain text", 0, 10},
		{`say "hi"`, 0, 4},
		{`a\b`, 0, 1},
		{"a<b", 0, 1},
		{"a>b", 0, 1},
		{"a&b", 0, 1},
		{"ab\x00", 0, 2},
		{"ab\x1f", 0, 2},
		{"ab\x20\x7f", 0, 4},
		{"\u00e9<", 0, 2},
		{"\u65e5\u672c\u8a9e\n", 0, 9},
		{"a\xffb", 0, 1},
		{"\xe2\x80", 0, 0},
		{"a\u2027b", 0, 5},
		{"a\u2028b", 0, 1},
		{"a\u2029b", 0, 1},
		{"a\ufffdb", 0, 5},
		{"xx<yy", 3, 5},
		{"xx<yy", 2, 2},
		{strings.Repeat("a", 40) + "<", 0, 40},
		{strings.Repeat("a", 40), 0, 40},
		{"a" + strings.Repeat("\u00e9", 20) + "\"", 0, 41},
	}
	for _, c := range cases {
		for _, prevRun := range []int{0, sparseSkip} {
			if got := firstEscape(c.s, c.from, prevRun); got != c.want {
				t.Errorf("firstEscape(%q, %d, %d) = %d, want %d", c.s, c.from, prevRun, got, c.want)
			}
		}
		if got := refFirstEscape(c.s, c.from); got != c.want {
			t.Errorf("refFirstEscape(%q, %d) = %d, want %d", c.s, c.from, got, c.want)
		}
	}
}

func TestEscapeReferenceRules(t *testing.T) {
	cases := []struct{ s, want string }{
		{"a<b", `a\u003cb`},
		{"a>b&c", `a\u003eb\u0026c`},
		{"\n\r\t\b\f", `\n\r\t\b\f`},
		{"\x01\x1f", `\u0001\u001f`},
		{"\u2028\u2029", `\u2028\u2029`},
		{`"\`, `\"\\`},
		{"\x7f\u00e9", "\x7f\u00e9"},
		{"a\xffb", "a" + invalidUTF8Escape + "b"},
	}
	for _, c := range cases {
		if got := refEscape(c.s); got != c.want {
			t.Errorf("refEscape(%q) = %q, want %q", c.s, got, c.want)
		}
		if got := string(AppendEscaped(nil, c.s)); got != c.want {
			t.Errorf("AppendEscaped(%q) = %q, want %q", c.s, got, c.want)
		}
	}
	if got, want := refQuoted("a<"), `\"a\\u003c\"`; got != want {
		t.Errorf("refQuoted(%q) = %q, want %q", "a<", got, want)
	}
}

func checkFirstEscape(t *testing.T, s string, from int) {
	t.Helper()
	want := refFirstEscape(s, from)
	for _, prevRun := range []int{0, sparseSkip} {
		if got := firstEscape(s, from, prevRun); got != want {
			t.Fatalf("firstEscape(%q, %d, %d) = %d, want %d", s, from, prevRun, got, want)
		}
	}
}

func checkEscapeLens(t *testing.T, s string) {
	t.Helper()
	want := refEscape(s)
	if got := string(AppendEscaped(nil, s)); got != want {
		t.Fatalf("AppendEscaped(%q) = %q, want %q", s, got, want)
	}
	if got := EscapedLen(s); got != len(want) {
		t.Fatalf("EscapedLen(%q) = %d, want %d", s, got, len(want))
	}
	wantQ := refQuoted(s)
	if got := string(AppendQuoted(nil, s)); got != wantQ {
		t.Fatalf("AppendQuoted(%q) = %q, want %q", s, got, wantQ)
	}
	if got := QuotedLen(s); got != len(wantQ) {
		t.Fatalf("QuotedLen(%q) = %d, want %d", s, got, len(wantQ))
	}
}

func diffFiller(n int) string {
	b := make([]byte, n)
	for j := range b {
		b[j] = diffPlainByte(j)
	}
	return string(b)
}

var gapRuns = []int{0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65}

var gapSequences = []string{"\"", "<", "\n", "\u00e9", "\u2028", "\xff", "\U0001f600"}

func forEachGapInput(seqs []string, visit func(body []byte)) {
	g, k := len(gapRuns), len(seqs)
	for n := range k * k * g * g * g {
		a, b := seqs[n%k], seqs[n/k%k]
		g1, g2, g3 := gapRuns[n/(k*k)%g], gapRuns[n/(k*k*g)%g], gapRuns[n/(k*k*g*g)]
		visit([]byte(diffFiller(g1) + a + diffFiller(g2) + b + diffFiller(g3)))
	}
}

func TestFirstEscapeDifferential(t *testing.T) {
	pad := strings.Repeat("p", 17)
	forEachPlacement(escapeSequences, func(body []byte) {
		for _, p := range []int{0, 3, 17} {
			s := pad[:p] + string(body)
			checkFirstEscape(t, s, p)
			if want := refFirstEscape(s, p); want < len(s) {
				checkFirstEscape(t, s, want+1)
			}
		}
	})
}

func TestEscapeLensDifferential(t *testing.T) {
	forEachPlacement(lenSequences, func(body []byte) {
		checkEscapeLens(t, string(body))
	})
}

func TestEscapeGapDifferential(t *testing.T) {
	forEachGapInput(gapSequences, func(body []byte) {
		s := string(body)
		checkEscapeLens(t, s)
		checkFirstEscape(t, s, 0)
	})
}
