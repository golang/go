// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics_test

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/doc"
	"go/doc/comment"
	"go/format"
	"go/parser"
	"go/token"
	"internal/diff"
	"os"
	"regexp"
	"runtime/metrics"
	"sort"
	"strings"
	"testing"
	_ "unsafe"
)

// Implemented in the runtime.
//
//go:linkname runtime_readMetricNames
func runtime_readMetricNames() []string

func TestNames(t *testing.T) {
	// Note that this regexp is promised in the package docs for Description. Do not change.
	r := regexp.MustCompile("^(?P<name>/[^:]+):(?P<unit>[^:*/]+(?:[*/][^:*/]+)*)$")
	all := metrics.All()
	for i, d := range all {
		if !r.MatchString(d.Name) {
			t.Errorf("name %q does not match regexp %#q", d.Name, r)
		}
		if i > 0 && all[i-1].Name >= all[i].Name {
			t.Fatalf("allDesc not sorted: %s ≥ %s", all[i-1].Name, all[i].Name)
		}
	}

	names := runtime_readMetricNames()
	sort.Strings(names)
	samples := make([]metrics.Sample, len(names))
	for i, name := range names {
		samples[i].Name = name
	}
	metrics.Read(samples)

	for _, d := range all {
		for len(samples) > 0 && samples[0].Name < d.Name {
			t.Errorf("%s: reported by runtime but not listed in All", samples[0].Name)
			samples = samples[1:]
		}
		if len(samples) == 0 || d.Name < samples[0].Name {
			t.Errorf("%s: listed in All but not reported by runtime", d.Name)
			continue
		}
		if samples[0].Value.Kind() != d.Kind {
			t.Errorf("%s: runtime reports %v but All reports %v", d.Name, samples[0].Value.Kind(), d.Kind)
		}
		samples = samples[1:]
	}
}

func wrap(prefix, text string, width int) string {
	doc := &comment.Doc{Content: []comment.Block{&comment.Paragraph{Text: []comment.Text{comment.Plain(text)}}}}
	pr := &comment.Printer{TextPrefix: prefix, TextWidth: width}
	return string(pr.Text(doc))
}

func formatDesc(t *testing.T) string {
	var b strings.Builder
	for i, d := range metrics.All() {
		if i > 0 {
			fmt.Fprintf(&b, "\n")
		}
		fmt.Fprintf(&b, "%s\n", d.Name)
		fmt.Fprintf(&b, "%s", wrap("\t", d.Description, 80-2*8))
	}
	return b.String()
}

var generate = flag.Bool("generate", false, "update doc.go for go generate")

func TestDocs(t *testing.T) {
	want := formatDesc(t)

	src, err := os.ReadFile("doc.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "doc.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	fdoc := f.Doc
	if fdoc == nil {
		t.Fatal("no doc comment in doc.go")
	}
	pkg, err := doc.NewFromFiles(fset, []*ast.File{f}, "runtime/metrics")
	if err != nil {
		t.Fatal(err)
	}
	if pkg.Doc == "" {
		t.Fatal("doc.NewFromFiles lost doc comment")
	}
	doc := new(comment.Parser).Parse(pkg.Doc)
	expectCode := false
	foundCode := false
	updated := false
	for _, block := range doc.Content {
		switch b := block.(type) {
		case *comment.Heading:
			expectCode = false
			if b.Text[0] == comment.Plain("Supported metrics") {
				expectCode = true
			}
		case *comment.Code:
			if expectCode {
				foundCode = true
				if b.Text != want {
					if !*generate {
						t.Fatalf("doc comment out of date; use go generate to rebuild\n%s", diff.Diff("old", []byte(b.Text), "want", []byte(want)))
					}
					b.Text = want
					updated = true
				}
			}
		}
	}

	if !foundCode {
		t.Fatalf("did not find Supported metrics list in doc.go")
	}
	if updated {
		fmt.Fprintf(os.Stderr, "go test -generate: writing new doc.go\n")
		var buf bytes.Buffer
		buf.Write(src[:fdoc.Pos()-f.FileStart])
		buf.WriteString("/*\n")
		buf.Write(new(comment.Printer).Comment(doc))
		buf.WriteString("*/")
		buf.Write(src[fdoc.End()-f.FileStart:])
		src, err := format.Source(buf.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile("doc.go", src, 0666); err != nil {
			t.Fatal(err)
		}
	} else if *generate {
		fmt.Fprintf(os.Stderr, "go test -generate: doc.go already up-to-date\n")
	}
}
