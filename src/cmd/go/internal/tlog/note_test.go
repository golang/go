// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tlog

import (
	"strings"
	"testing"
)

func TestFormatTree(t *testing.T) {
	n := int64(123456789012)
	h := RecordHash([]byte("hello world"))
	golden := "go.sum database tree\n123456789012\nTszzRgjTG6xce+z2AG31kAXYKBgQVtCSCE40HmuwBb0=\n"
	b := FormatTree(Tree{n, h})
	if string(b) != golden {
		t.Errorf("FormatTree(...) = %q, want %q", b, golden)
	}
}

func TestParseTree(t *testing.T) {
	in := "go.sum database tree\n123456789012\nTszzRgjTG6xce+z2AG31kAXYKBgQVtCSCE40HmuwBb0=\n"
	goldH := RecordHash([]byte("hello world"))
	goldN := int64(123456789012)
	tree, err := ParseTree([]byte(in))
	if tree.N != goldN || tree.Hash != goldH || err != nil {
		t.Fatalf("ParseTree(...) = Tree{%d, %v}, %v, want Tree{%d, %v}, nil", tree.N, tree.Hash, err, goldN, goldH)
	}

	// Check invalid trees.
	var badTrees = []string{
		"not-" + in,
		"go.sum database tree\n0xabcdef\nTszzRgjTG6xce+z2AG31kAXYKBgQVtCSCE40HmuwBb0=\n",
		"go.sum database tree\n123456789012\nTszzRgjTG6xce+z2AG31kAXYKBgQVtCSCE40HmuwBTOOBIG=\n",
	}
	for _, bad := range badTrees {
		_, err := ParseTree([]byte(bad))
		if err == nil {
			t.Fatalf("ParseTree(%q) succeeded, want failure", in)
		}
	}

	// Check junk on end is ignored.
	var goodTrees = []string{
		in + "JOE",
		in + "JOE\n",
		in + strings.Repeat("JOE\n", 1000),
	}
	for _, good := range goodTrees {
		_, err := ParseTree([]byte(good))
		if tree.N != goldN || tree.Hash != goldH || err != nil {
			t.Fatalf("ParseTree(...+%q) = Tree{%d, %v}, %v, want Tree{%d, %v}, nil", good[len(in):], tree.N, tree.Hash, err, goldN, goldH)
		}
	}
}

func TestFormatRecord(t *testing.T) {
	id := int64(123456789012)
	text := "hello, world\n"
	golden := "123456789012\nhello, world\n\n"
	msg, err := FormatRecord(id, []byte(text))
	if err != nil {
		t.Fatalf("FormatRecord: %v", err)
	}
	if string(msg) != golden {
		t.Fatalf("FormatRecord(...) = %q, want %q", msg, golden)
	}

	var badTexts = []string{
		"",
		"hello\nworld",
		"hello\n\nworld\n",
		"hello\x01world\n",
	}
	for _, bad := range badTexts {
		msg, err := FormatRecord(id, []byte(bad))
		if err == nil {
			t.Errorf("FormatRecord(id, %q) = %q, want error", bad, msg)
		}
	}
}

func TestParseRecord(t *testing.T) {
	in := "123456789012\nhello, world\n\njunk on end\x01\xff"
	goldID := int64(123456789012)
	goldText := "hello, world\n"
	goldRest := "junk on end\x01\xff"
	id, text, rest, err := ParseRecord([]byte(in))
	if id != goldID || string(text) != goldText || string(rest) != goldRest || err != nil {
		t.Fatalf("ParseRecord(%q) = %d, %q, %q, %v, want %d, %q, %q, nil", in, id, text, rest, err, goldID, goldText, goldRest)
	}

	in = "123456789012\nhello, world\n\n"
	id, text, rest, err = ParseRecord([]byte(in))
	if id != goldID || string(text) != goldText || len(rest) != 0 || err != nil {
		t.Fatalf("ParseRecord(%q) = %d, %q, %q, %v, want %d, %q, %q, nil", in, id, text, rest, err, goldID, goldText, "")
	}
	if rest == nil {
		t.Fatalf("ParseRecord(%q): rest = []byte(nil), want []byte{}", in)
	}

	// Check invalid records.
	var badRecords = []string{
		"not-" + in,
		"123\nhello\x01world\n\n",
		"123\nhello\xffworld\n\n",
		"123\nhello world\n",
		"0x123\nhello world\n\n",
	}
	for _, bad := range badRecords {
		id, text, rest, err := ParseRecord([]byte(bad))
		if err == nil {
			t.Fatalf("ParseRecord(%q) = %d, %q, %q, nil, want error", in, id, text, rest)
		}
	}
}
