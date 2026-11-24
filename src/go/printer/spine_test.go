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

// spineLinks are the links of left-leaning expression chains
// whose position is that of their leftmost operand.
var spineLinks = []string{
	".b",
	".b()",
	".\n\tb(c)",
	"[0]",
	"[i, j]",
	"[1:2]",
	".(T)",
	".b( /* c */ x).\n\t// d\n\te[0]",
}

func spineSource(link string, n int) string {
	return "package p\n\nvar x = a" + strings.Repeat(link, n) + "\n"
}

func TestSpineChain(t *testing.T) {
	for _, link := range spineLinks {
		src := spineSource(link, 100)
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
			t.Errorf("chain of %q is not formatted as is:\n%s", link, got)
		}
	}
}

func BenchmarkSpineChain(b *testing.B) {
	for _, link := range []string{".b()", "[0]"} {
		for _, n := range []int{1000, 4000, 16000} {
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "", spineSource(link, n), parser.ParseComments|parser.SkipObjectResolution)
			if err != nil {
				b.Fatal(err)
			}
			b.Run(fmt.Sprintf("%s/%d", link, n), func(b *testing.B) {
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
}
