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
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

func pipeErr(err error) io.Reader {
	pr, pw := io.Pipe()
	pw.CloseWithError(err)
	return pr
}

func readDat(filename string, c chan io.Reader) {
	defer close(c)
	f, err := os.Open("testdata/webkit/" + filename)
	if err != nil {
		c <- pipeErr(err)
		return
	}
	defer f.Close()

	// Loop through the lines of the file. Each line beginning with "#" denotes
	// a new section, which is returned as a separate io.Reader.
	r := bufio.NewReader(f)
	var pw *io.PipeWriter
	for {
		line, err := r.ReadSlice('\n')
		if err != nil {
			if pw != nil {
				pw.CloseWithError(err)
				pw = nil
			} else {
				c <- pipeErr(err)
			}
			return
		}
		if len(line) == 0 {
			continue
		}
		if line[0] == '#' {
			if pw != nil {
				pw.Close()
			}
			var pr *io.PipeReader
			pr, pw = io.Pipe()
			c <- pr
			continue
		}
		if line[0] != '|' {
			// Strip the trailing '\n'.
			line = line[:len(line)-1]
		}
		if pw != nil {
			if _, err := pw.Write(line); err != nil {
				pw.CloseWithError(err)
				pw = nil
			}
		}
	}
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
		fmt.Fprintf(w, "<%s>", n.Data)
		for _, a := range n.Attr {
			io.WriteString(w, "\n")
			dumpIndent(w, level+1)
			fmt.Fprintf(w, `%s="%s"`, a.Key, a.Val)
		}
	case TextNode:
		fmt.Fprintf(w, "%q", n.Data)
	case CommentNode:
		fmt.Fprintf(w, "<!-- %s -->", n.Data)
	case DoctypeNode:
		fmt.Fprintf(w, "<!DOCTYPE %s>", n.Data)
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
		{"tests1.dat", -1},
		{"tests2.dat", 43},
		{"tests3.dat", 0},
	}
	for _, tf := range testFiles {
		rc := make(chan io.Reader)
		go readDat(tf.filename, rc)
		for i := 0; i != tf.n; i++ {
			// Parse the #data section.
			dataReader := <-rc
			if dataReader == nil {
				break
			}
			b, err := ioutil.ReadAll(dataReader)
			if err != nil {
				t.Fatal(err)
			}
			text := string(b)
			doc, err := Parse(strings.NewReader(text))
			if err != nil {
				t.Fatal(err)
			}
			got, err := dump(doc)
			if err != nil {
				t.Fatal(err)
			}
			// Skip the #error section.
			if _, err := io.Copy(ioutil.Discard, <-rc); err != nil {
				t.Fatal(err)
			}
			// Compare the parsed tree to the #document section.
			b, err = ioutil.ReadAll(<-rc)
			if err != nil {
				t.Fatal(err)
			}
			if want := string(b); got != want {
				t.Errorf("%s test #%d %q, got vs want:\n----\n%s----\n%s----", tf.filename, i, text, got, want)
				continue
			}
			if renderTestBlacklist[text] {
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
		// Drain any untested cases for the test file.
		for r := range rc {
			if _, err := ioutil.ReadAll(r); err != nil {
				t.Fatal(err)
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
