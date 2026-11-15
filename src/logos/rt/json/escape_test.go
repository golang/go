// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"strings"
	"testing"
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
