// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printer

import (
	"bytes"
	"fmt"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

func chainSource(n int) string {
	return "package p\n\nvar x = " + strings.Repeat("a +\n\t", n) + "a\n"
}

func TestBinaryChain(t *testing.T) {
	src := chainSource(100)
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := (&Config{Mode: UseSpaces | TabIndent, Tabwidth: 8}).Fprint(&buf, fset, f); err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); got != src {
		t.Errorf("chain is not formatted as is:\n%s", got)
	}
}

func BenchmarkBinaryChain(b *testing.B) {
	for _, n := range []int{1000, 2000, 4000, 8000} {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "", chainSource(n), parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				var buf bytes.Buffer
				if err := (&Config{Mode: UseSpaces | TabIndent, Tabwidth: 8}).Fprint(&buf, fset, f); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
