// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

var updateLogs = flag.Bool("update-logs", false, "Update the log files that show the test results")

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
	var b bytes.Buffer
	for _, child := range n.Child {
		if err := dumpLevel(&b, child, 0); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

const testDataDir = "testdata/webkit/"
const testLogDir = "testlogs/"

type parseTestResult int

const (
	// parseTestFailed indicates that an error occurred during parsing or that
	// the parse tree did not match the expected result.
	parseTestFailed parseTestResult = iota
	// parseTestParseOnly indicates that the first stage of the test (parsing)
	// passed, but rendering and re-parsing did not.
	parseTestParseOnly
	// parseTestPassed indicates that both stages of the test passed.
	parseTestPassed
)

func (r parseTestResult) String() string {
	switch r {
	case parseTestFailed:
		return "FAIL"
	case parseTestParseOnly:
		return "PARSE"
	case parseTestPassed:
		return "PASS"
	}
	return "invalid parseTestResult value"
}

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

		logName := testLogDir + tf[len(testDataDir):] + ".log"
		var lf *os.File
		var lbr *bufio.Reader
		if *updateLogs {
			lf, err = os.Create(logName)
		} else {
			lf, err = os.Open(logName)
			lbr = bufio.NewReader(lf)
		}
		if err != nil {
			t.Fatal(err)
		}
		defer lf.Close()

		for i := 0; ; i++ {
			text, want, context, err := readParseTest(r)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}

			var expectedResult parseTestResult
			if !*updateLogs {
				var expectedText, expectedResultString string
				_, err = fmt.Fscanf(lbr, "%s %q\n", &expectedResultString, &expectedText)
				if err != nil {
					t.Fatal(err)
				}
				if expectedText != text {
					t.Fatalf("Log does not match tests: log has %q, tests have %q", expectedText, text)
				}
				switch expectedResultString {
				case "FAIL":
					// Skip this test.
					continue
				case "PARSE":
					expectedResult = parseTestParseOnly
				case "PASS":
					expectedResult = parseTestPassed
				default:
					t.Fatalf("Log has invalid test result: %q", expectedResultString)
				}
			}

			result, err := testParseCase(text, want, context)

			if *updateLogs {
				fmt.Fprintf(lf, "%s %q\n", result, text)
			} else if result < expectedResult {
				t.Errorf("%s test #%d %q, %s", tf, i, text, err)
			}
		}
	}
}

// testParseCase tests one test case from the test files. It returns a 
// parseTestResult indicating how much of the test passed. If the result
// is not parseTestPassed, it also returns an error that explains the failure.
// text is the HTML to be parsed, want is a dump of the correct parse tree,
// and context is the name of the context node, if any.
func testParseCase(text, want, context string) (result parseTestResult, err error) {
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
			return parseTestFailed, err
		}
	} else {
		contextNode := &Node{
			Type: ElementNode,
			Data: context,
		}
		nodes, err := ParseFragment(strings.NewReader(text), contextNode)
		if err != nil {
			return parseTestFailed, err
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
		return parseTestFailed, err
	}
	// Compare the parsed tree to the #document section.
	if got != want {
		return parseTestFailed, fmt.Errorf("got vs want:\n----\n%s----\n%s----", got, want)
	}

	if renderTestBlacklist[text] || context != "" {
		return parseTestPassed, nil
	}

	// Set result so that if a panic occurs during the render and re-parse
	// the calling function will know that the parsing phase was successful.
	result = parseTestParseOnly

	// Check that rendering and re-parsing results in an identical tree.
	pr, pw := io.Pipe()
	go func() {
		pw.CloseWithError(Render(pw, doc))
	}()
	doc1, err := Parse(pr)
	if err != nil {
		return parseTestParseOnly, err
	}
	got1, err := dump(doc1)
	if err != nil {
		return parseTestParseOnly, err
	}
	if got != got1 {
		return parseTestParseOnly, fmt.Errorf("got vs got1:\n----\n%s----\n%s----", got, got1)
	}

	return parseTestPassed, nil
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
