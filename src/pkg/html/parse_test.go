// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
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
	text = strings.TrimRight(string(b), "\n")
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
	for {
		line, err = r.ReadSlice('\n')
		if err != nil && err != io.EOF {
			return "", "", "", err
		}
		if len(line) == 0 || len(line) == 1 && line[0] == '\n' {
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
		attr := n.Attr
		if len(attr) == 2 && attr[0].Namespace == "xml" && attr[1].Namespace == "xlink" {
			// Some of the test cases in tests10.dat change the order of adjusted
			// foreign attributes, but that behavior is not in the spec, and could
			// simply be an implementation detail of html5lib's python map ordering.
			attr[0], attr[1] = attr[1], attr[0]
		}
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
	for _, c := range n.Child {
		if err := dumpLevel(w, c, level+1); err != nil {
			return err
		}
	}
	return nil
}

func dump(n *Node) (string, error) {
	if n == nil || len(n.Child) == 0 {
		return "", nil
	}
	b := bytes.NewBuffer(nil)
	for _, child := range n.Child {
		if err := dumpLevel(b, child, 0); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

func TestParser(t *testing.T) {
	testFiles := []struct {
		filename string
		// n is the number of test cases to run from that file.
		// -1 means all test cases.
		n int
	}{
		// TODO(nigeltao): Process all the test cases from all the .dat files.
		{"adoption01.dat", -1},
		{"doctype01.dat", -1},
		{"tests1.dat", -1},
		{"tests2.dat", -1},
		{"tests3.dat", -1},
		{"tests4.dat", -1},
		{"tests5.dat", -1},
		{"tests6.dat", -1},
		{"tests10.dat", 35},
	}
	for _, tf := range testFiles {
		f, err := os.Open("testdata/webkit/" + tf.filename)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		r := bufio.NewReader(f)
		for i := 0; i != tf.n; i++ {
			text, want, context, err := readParseTest(r)
			if err == io.EOF && tf.n == -1 {
				break
			}
			if err != nil {
				t.Fatal(err)
			}

			var doc *Node
			if context == "" {
				doc, err = Parse(strings.NewReader(text))
				if err != nil {
					t.Fatal(err)
				}
			} else {
				contextNode := &Node{
					Type: ElementNode,
					Data: context,
				}
				nodes, err := ParseFragment(strings.NewReader(text), contextNode)
				if err != nil {
					t.Fatal(err)
				}
				doc = &Node{
					Type: DocumentNode,
				}
				for _, n := range nodes {
					doc.Add(n)
				}
			}

			got, err := dump(doc)
			if err != nil {
				t.Fatal(err)
			}
			// Compare the parsed tree to the #document section.
			if got != want {
				t.Errorf("%s test #%d %q, got vs want:\n----\n%s----\n%s----", tf.filename, i, text, got, want)
				continue
			}
			if renderTestBlacklist[text] || context != "" {
				continue
			}
			// Check that rendering and re-parsing results in an identical tree.
			pr, pw := io.Pipe()
			go func() {
				pw.CloseWithError(Render(pw, doc))
			}()
			doc1, err := Parse(pr)
			if err != nil {
				t.Fatal(err)
			}
			got1, err := dump(doc1)
			if err != nil {
				t.Fatal(err)
			}
			if got != got1 {
				t.Errorf("%s test #%d %q, got vs got1:\n----\n%s----\n%s----", tf.filename, i, text, got, got1)
				continue
			}
		}
	}
}

// Some test input result in parse trees are not 'well-formed' despite
// following the HTML5 recovery algorithms. Rendering and re-parsing such a
// tree will not result in an exact clone of that tree. We blacklist such
// inputs from the render test.
var renderTestBlacklist = map[string]bool{
	// The second <a> will be reparented to the first <table>'s parent. This
	// results in an <a> whose parent is an <a>, which is not 'well-formed'.
	`<a><table><td><a><table></table><a></tr><a></table><b>X</b>C<a>Y`: true,
	// More cases of <a> being reparented:
	`<a href="blah">aba<table><a href="foo">br<tr><td></td></tr>x</table>aoe`: true,
	`<a><table><a></table><p><a><div><a>`:                                     true,
	`<a><table><td><a><table></table><a></tr><a></table><a>`:                  true,
	// A <plaintext> element is reparented, putting it before a table.
	// A <plaintext> element can't have anything after it in HTML.
	`<table><plaintext><td>`: true,
}
