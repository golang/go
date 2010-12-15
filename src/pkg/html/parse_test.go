// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

type devNull struct{}

func (devNull) Write(p []byte) (int, os.Error) {
	return len(p), nil
}

func pipeErr(err os.Error) io.Reader {
	pr, pw := io.Pipe()
	pw.CloseWithError(err)
	return pr
}

func readDat(filename string, c chan io.Reader) {
	f, err := os.Open("testdata/webkit/"+filename, os.O_RDONLY, 0600)
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

func dumpLevel(w io.Writer, n *Node, level int) os.Error {
	io.WriteString(w, "| ")
	for i := 0; i < level; i++ {
		io.WriteString(w, "  ")
	}
	switch n.Type {
	case ErrorNode:
		return os.NewError("unexpected ErrorNode")
	case DocumentNode:
		return os.NewError("unexpected DocumentNode")
	case ElementNode:
		fmt.Fprintf(w, "<%s>", EscapeString(n.Data))
	case TextNode:
		fmt.Fprintf(w, "%q", EscapeString(n.Data))
	case CommentNode:
		return os.NewError("COMMENT")
	default:
		return os.NewError("unknown node type")
	}
	io.WriteString(w, "\n")
	for _, c := range n.Child {
		if err := dumpLevel(w, c, level+1); err != nil {
			return err
		}
	}
	return nil
}

func dump(n *Node) (string, os.Error) {
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
	// TODO(nigeltao): Process all the .dat files, not just the first one.
	filenames := []string{
		"tests1.dat",
	}
	for _, filename := range filenames {
		rc := make(chan io.Reader)
		go readDat(filename, rc)
		// TODO(nigeltao): Process all test cases, not just a subset.
		for i := 0; i < 22; i++ {
			// Parse the #data section.
			b, err := ioutil.ReadAll(<-rc)
			if err != nil {
				t.Fatal(err)
			}
			text := string(b)
			doc, err := Parse(strings.NewReader(text))
			if err != nil {
				t.Fatal(err)
			}
			actual, err := dump(doc)
			if err != nil {
				t.Fatal(err)
			}
			// Skip the #error section.
			if _, err := io.Copy(devNull{}, <-rc); err != nil {
				t.Fatal(err)
			}
			// Compare the parsed tree to the #document section.
			b, err = ioutil.ReadAll(<-rc)
			if err != nil {
				t.Fatal(err)
			}
			expected := string(b)
			if actual != expected {
				t.Errorf("%s test #%d %q, actual vs expected:\n----\n%s----\n%s----", filename, i, text, actual, expected)
			}
		}
	}
}
