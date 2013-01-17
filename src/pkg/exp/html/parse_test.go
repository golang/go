// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bufio"
	"bytes"
	"errors"
	"exp/html/atom"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// readParseTest reads a single test case from r.
func readParseTest(r *bufio.Reader) (text, want, context string, err error) {
	line, err := r.ReadSlice('\n')
	if err != nil {
		return "", "", "", err
	}
	var b []byte

	// Read the HTML.
	if string(line) != "#data\n" {
		return "", "", "", fmt.Errorf(`got %q want "#data\n"`, line)
	}
	for {
		line, err = r.ReadSlice('\n')
		if err != nil {
			return "", "", "", err
		}
		if line[0] == '#' {
			break
		}
		b = append(b, line...)
	}
	text = string(b)
	if strings.HasSuffix(text, "\n") {
		text = text[:len(text)-1]
	}
	b = b[:0]

	// Skip the error list.
	if string(line) != "#errors\n" {
		return "", "", "", fmt.Errorf(`got %q want "#errors\n"`, line)
	}
	for {
		line, err = r.ReadSlice('\n')
		if err != nil {
			return "", "", "", err
		}
		if line[0] == '#' {
			break
		}
	}

	if string(line) == "#document-fragment\n" {
		line, err = r.ReadSlice('\n')
		if err != nil {
			return "", "", "", err
		}
		context = strings.TrimSpace(string(line))
		line, err = r.ReadSlice('\n')
		if err != nil {
			return "", "", "", err
		}
	}

	// Read the dump of what the parse tree should be.
	if string(line) != "#document\n" {
		return "", "", "", fmt.Errorf(`got %q want "#document\n"`, line)
	}
	inQuote := false
	for {
		line, err = r.ReadSlice('\n')
		if err != nil && err != io.EOF {
			return "", "", "", err
		}
		trimmed := bytes.Trim(line, "| \n")
		if len(trimmed) > 0 {
			if line[0] == '|' && trimmed[0] == '"' {
				inQuote = true
			}
			if trimmed[len(trimmed)-1] == '"' && !(line[0] == '|' && len(trimmed) == 1) {
				inQuote = false
			}
		}
		if len(line) == 0 || len(line) == 1 && line[0] == '\n' && !inQuote {
			break
		}
		b = append(b, line...)
	}
	return text, string(b), context, nil
}

func dumpIndent(w io.Writer, level int) {
	io.WriteString(w, "| ")
	for i := 0; i < level; i++ {
		io.WriteString(w, "  ")
	}
}

type sortedAttributes []Attribute

func (a sortedAttributes) Len() int {
	return len(a)
}

func (a sortedAttributes) Less(i, j int) bool {
	if a[i].Namespace != a[j].Namespace {
		return a[i].Namespace < a[j].Namespace
	}
	return a[i].Key < a[j].Key
}

func (a sortedAttributes) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func dumpLevel(w io.Writer, n *Node, level int) error {
	dumpIndent(w, level)
	switch n.Type {
	case ErrorNode:
		return errors.New("unexpected ErrorNode")
	case DocumentNode:
		return errors.New("unexpected DocumentNode")
	case ElementNode:
		if n.Namespace != "" {
			fmt.Fprintf(w, "<%s %s>", n.Namespace, n.Data)
		} else {
			fmt.Fprintf(w, "<%s>", n.Data)
		}
		attr := sortedAttributes(n.Attr)
		sort.Sort(attr)
		for _, a := range attr {
			io.WriteString(w, "\n")
			dumpIndent(w, level+1)
			if a.Namespace != "" {
				fmt.Fprintf(w, `%s %s="%s"`, a.Namespace, a.Key, a.Val)
			} else {
				fmt.Fprintf(w, `%s="%s"`, a.Key, a.Val)
			}
		}
	case TextNode:
		fmt.Fprintf(w, `"%s"`, n.Data)
	case CommentNode:
		fmt.Fprintf(w, "<!-- %s -->", n.Data)
	case DoctypeNode:
		fmt.Fprintf(w, "<!DOCTYPE %s", n.Data)
		if n.Attr != nil {
			var p, s string
			for _, a := range n.Attr {
				switch a.Key {
				case "public":
					p = a.Val
				case "system":
					s = a.Val
				}
			}
			if p != "" || s != "" {
				fmt.Fprintf(w, ` "%s"`, p)
				fmt.Fprintf(w, ` "%s"`, s)
			}
		}
		io.WriteString(w, ">")
	case scopeMarkerNode:
		return errors.New("unexpected scopeMarkerNode")
	default:
		return errors.New("unknown node type")
	}
	io.WriteString(w, "\n")
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if err := dumpLevel(w, c, level+1); err != nil {
			return err
		}
	}
	return nil
}

func dump(n *Node) (string, error) {
	if n == nil || n.FirstChild == nil {
		return "", nil
	}
	var b bytes.Buffer
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if err := dumpLevel(&b, c, 0); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

const testDataDir = "testdata/webkit/"

func TestParser(t *testing.T) {
	testFiles, err := filepath.Glob(testDataDir + "*.dat")
	if err != nil {
		t.Fatal(err)
	}
	for _, tf := range testFiles {
		f, err := os.Open(tf)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		r := bufio.NewReader(f)

		for i := 0; ; i++ {
			text, want, context, err := readParseTest(r)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}

			err = testParseCase(text, want, context)

			if err != nil {
				t.Errorf("%s test #%d %q, %s", tf, i, text, err)
			}
		}
	}
}

// testParseCase tests one test case from the test files. If the test does not
// pass, it returns an error that explains the failure.
// text is the HTML to be parsed, want is a dump of the correct parse tree,
// and context is the name of the context node, if any.
func testParseCase(text, want, context string) (err error) {
	defer func() {
		if x := recover(); x != nil {
			switch e := x.(type) {
			case error:
				err = e
			default:
				err = fmt.Errorf("%v", e)
			}
		}
	}()

	var doc *Node
	if context == "" {
		doc, err = Parse(strings.NewReader(text))
		if err != nil {
			return err
		}
	} else {
		contextNode := &Node{
			Type:     ElementNode,
			DataAtom: atom.Lookup([]byte(context)),
			Data:     context,
		}
		nodes, err := ParseFragment(strings.NewReader(text), contextNode)
		if err != nil {
			return err
		}
		doc = &Node{
			Type: DocumentNode,
		}
		for _, n := range nodes {
			doc.AppendChild(n)
		}
	}

	if err := checkTreeConsistency(doc); err != nil {
		return err
	}

	got, err := dump(doc)
	if err != nil {
		return err
	}
	// Compare the parsed tree to the #document section.
	if got != want {
		return fmt.Errorf("got vs want:\n----\n%s----\n%s----", got, want)
	}

	if renderTestBlacklist[text] || context != "" {
		return nil
	}

	// Check that rendering and re-parsing results in an identical tree.
	pr, pw := io.Pipe()
	go func() {
		pw.CloseWithError(Render(pw, doc))
	}()
	doc1, err := Parse(pr)
	if err != nil {
		return err
	}
	got1, err := dump(doc1)
	if err != nil {
		return err
	}
	if got != got1 {
		return fmt.Errorf("got vs got1:\n----\n%s----\n%s----", got, got1)
	}

	return nil
}

// Some test input result in parse trees are not 'well-formed' despite
// following the HTML5 recovery algorithms. Rendering and re-parsing such a
// tree will not result in an exact clone of that tree. We blacklist such
// inputs from the render test.
var renderTestBlacklist = map[string]bool{
	// The second <a> will be reparented to the first <table>'s parent. This
	// results in an <a> whose parent is an <a>, which is not 'well-formed'.
	`<a><table><td><a><table></table><a></tr><a></table><b>X</b>C<a>Y`: true,
	// The same thing with a <p>:
	`<p><table></p>`: true,
	// More cases of <a> being reparented:
	`<a href="blah">aba<table><a href="foo">br<tr><td></td></tr>x</table>aoe`: true,
	`<a><table><a></table><p><a><div><a>`:                                     true,
	`<a><table><td><a><table></table><a></tr><a></table><a>`:                  true,
	// A similar reparenting situation involving <nobr>:
	`<!DOCTYPE html><body><b><nobr>1<table><nobr></b><i><nobr>2<nobr></i>3`: true,
	// A <plaintext> element is reparented, putting it before a table.
	// A <plaintext> element can't have anything after it in HTML.
	`<table><plaintext><td>`:                                   true,
	`<!doctype html><table><plaintext></plaintext>`:            true,
	`<!doctype html><table><tbody><plaintext></plaintext>`:     true,
	`<!doctype html><table><tbody><tr><plaintext></plaintext>`: true,
	// A form inside a table inside a form doesn't work either.
	`<!doctype html><form><table></form><form></table></form>`: true,
	// A script that ends at EOF may escape its own closing tag when rendered.
	`<!doctype html><script><!--<script `:          true,
	`<!doctype html><script><!--<script <`:         true,
	`<!doctype html><script><!--<script <a`:        true,
	`<!doctype html><script><!--<script </`:        true,
	`<!doctype html><script><!--<script </s`:       true,
	`<!doctype html><script><!--<script </script`:  true,
	`<!doctype html><script><!--<script </scripta`: true,
	`<!doctype html><script><!--<script -`:         true,
	`<!doctype html><script><!--<script -a`:        true,
	`<!doctype html><script><!--<script -<`:        true,
	`<!doctype html><script><!--<script --`:        true,
	`<!doctype html><script><!--<script --a`:       true,
	`<!doctype html><script><!--<script --<`:       true,
	`<script><!--<script `:                         true,
	`<script><!--<script <a`:                       true,
	`<script><!--<script </script`:                 true,
	`<script><!--<script </scripta`:                true,
	`<script><!--<script -`:                        true,
	`<script><!--<script -a`:                       true,
	`<script><!--<script --`:                       true,
	`<script><!--<script --a`:                      true,
	`<script><!--<script <`:                        true,
	`<script><!--<script </`:                       true,
	`<script><!--<script </s`:                      true,
	// Reconstructing the active formatting elements results in a <plaintext>
	// element that contains an <a> element.
	`<!doctype html><p><a><plaintext>b`: true,
}

func TestNodeConsistency(t *testing.T) {
	// inconsistentNode is a Node whose DataAtom and Data do not agree.
	inconsistentNode := &Node{
		Type:     ElementNode,
		DataAtom: atom.Frameset,
		Data:     "table",
	}
	_, err := ParseFragment(strings.NewReader("<p>hello</p>"), inconsistentNode)
	if err == nil {
		t.Errorf("got nil error, want non-nil")
	}
}

func BenchmarkParser(b *testing.B) {
	buf, err := ioutil.ReadFile("testdata/go1.html")
	if err != nil {
		b.Fatalf("could not read testdata/go1.html: %v", err)
	}
	b.SetBytes(int64(len(buf)))
	runtime.GC()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Parse(bytes.NewBuffer(buf))
	}
}
