// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"log"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
)

func init() {
	log.SetFlags(log.Lshortfile)
}

type tparse struct {
	marked string   // ^ shows where to ask for completions. (The user just typed the following character.)
	wanted []string // expected completions
}

// Test completions in templates that parse enough (if completion needs symbols)
func TestParsed(t *testing.T) {
	var tests = []tparse{
		{"{{^if}}", []string{"index", "if"}},
		{"{{if .}}{{^e {{end}}", []string{"eq", "end}}", "else", "end"}},
		{"{{foo}}{{^f", []string{"foo"}},
		{"{{^$}}", []string{"$"}},
		{"{{$x:=4}}{{^$", []string{"$x"}},
		{"{{$x:=4}}{{$^ ", []string{}},
		{"{{len .Modified}}{{^.Mo", []string{"Modified"}},
		{"{{len .Modified}}{{.m^f", []string{"Modified"}},
		{"{{^$ }}", []string{"$"}},
		{"{{$a =3}}{{^$", []string{"$a"}},
		// .two is not good here: fix someday
		{`{{.Modified}}{{^.{{if $.one.two}}xxx{{end}}`, []string{"Modified", "one", "two"}},
		{`{{.Modified}}{{.^o{{if $.one.two}}xxx{{end}}`, []string{"one"}},
		{"{{.Modiifed}}{{.one.^t{{if $.one.two}}xxx{{end}}", []string{"two"}},
		{`{{block "foo" .}}{{^i`, []string{"index", "if"}},
		{"{{i^n{{Internal}}", []string{"index", "Internal", "if"}},
		// simple number has no completions
		{"{{4^e", []string{}},
		// simple string has no completions
		{"{{`^e", []string{}},
		{"{{`No ^i", []string{}}, // example of why go/scanner is used
		{"{{xavier}}{{12. ^x", []string{"xavier"}},
	}
	for _, tx := range tests {
		c := testCompleter(t, tx)
		ans, err := c.complete()
		if err != nil {
			t.Fatal(err)
		}
		var v []string
		for _, a := range ans.Items {
			v = append(v, a.Label)
		}
		if len(v) != len(tx.wanted) {
			t.Errorf("%q: got %v, wanted %v", tx.marked, v, tx.wanted)
			continue
		}
		sort.Strings(tx.wanted)
		sort.Strings(v)
		for i := 0; i < len(v); i++ {
			if tx.wanted[i] != v[i] {
				t.Errorf("%q at %d: got %v, wanted %v", tx.marked, i, v, tx.wanted)
				break
			}
		}
	}
}

func testCompleter(t *testing.T, tx tparse) *completer {
	t.Helper()
	col := strings.Index(tx.marked, "^") + 1
	offset := strings.LastIndex(tx.marked[:col], string(Left))
	if offset < 0 {
		t.Fatalf("no {{ before ^: %q", tx.marked)
	}
	buf := strings.Replace(tx.marked, "^", "", 1)
	p := parseBuffer([]byte(buf))
	if p.ParseErr != nil {
		log.Printf("%q: %v", tx.marked, p.ParseErr)
	}
	syms := make(map[string]symbol)
	filterSyms(syms, p.symbols)
	c := &completer{
		p:      p,
		pos:    protocol.Position{Line: 0, Character: uint32(col)},
		offset: offset + len(Left),
		ctx:    protocol.CompletionContext{TriggerKind: protocol.Invoked},
		syms:   syms,
	}
	return c
}
