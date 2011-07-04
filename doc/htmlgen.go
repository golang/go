// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// If --html is set, process plain text into HTML.
//	- h2's are made from lines followed by a line "----\n"
//	- tab-indented blocks become <pre> blocks with the first tab deleted
//	- blank lines become <p> marks (except inside <pre> tags)
//	- "quoted strings" become <code>quoted strings</code>

// Lines beginning !src define pieces of program source to be
// extracted from other files and injected as <pre> blocks.
// The syntax is simple: 1, 2, or 3 space-separated arguments:
//
// Whole file:
//	!src foo.go
// One line (here the signature of main):
//	!src foo.go /^func.main/
// Block of text, determined by start and end (here the body of main):
// !src foo.go /^func.main/ /^}/
//
// Patterns can be /regular.expression/, a decimal number, or $
// to signify the end of the file.
// TODO: the regular expression cannot contain spaces; does this matter?

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
	"template"
)

var (
	html = flag.Bool("html", true, "process text into HTML")
)

var (
	// lines holds the input and is reworked in place during processing.
	lines = make([][]byte, 0, 20000)

	empty   = []byte("")
	newline = []byte("\n")
	tab     = []byte("\t")
	quote   = []byte(`"`)
	indent  = []byte("    ")

	sectionMarker = []byte("----\n")
	preStart      = []byte("<pre>")
	preEnd        = []byte("</pre>\n")
	pp            = []byte("<p>\n")

	srcPrefix = []byte("!src")
)

func main() {
	flag.Parse()
	read()
	programs()
	if *html {
		headings()
		coalesce(preStart, foldPre)
		coalesce(tab, foldTabs)
		paragraphs()
		quotes()
	}
	write()
}

// read turns standard input into a slice of lines.
func read() {
	b := bufio.NewReader(os.Stdin)
	for {
		line, err := b.ReadBytes('\n')
		if err == os.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		lines = append(lines, line)
	}
}

// write puts the result on standard output.
func write() {
	b := bufio.NewWriter(os.Stdout)
	for _, line := range lines {
		b.Write(expandTabs(line))
	}
	b.Flush()
}

// programs injects source code from !src invocations.
func programs() {
	nlines := make([][]byte, 0, len(lines)*3/2)
	for _, line := range lines {
		if bytes.HasPrefix(line, srcPrefix) {
			line = trim(line)[len(srcPrefix):]
			prog := srcCommand(string(line))
			if *html {
				nlines = append(nlines, []byte(fmt.Sprintf("<pre><!--%s\n-->", line)))
			}
			for _, l := range prog {
				nlines = append(nlines, htmlEscape(l))
			}
			if *html {
				nlines = append(nlines, preEnd)
			}
		} else {
			nlines = append(nlines, line)
		}
	}
	lines = nlines
}

// srcCommand processes one !src invocation.
func srcCommand(command string) [][]byte {
	// TODO: quoted args so we can have 'a b'?
	args := strings.Fields(command)
	if len(args) == 0 || len(args) > 3 {
		log.Fatal("bad syntax for src command: %s", command)
	}
	file := args[0]
	lines := bytes.SplitAfter(readFile(file), newline)
	// File plus zero args: whole file:
	//	!src file.go
	if len(args) == 1 {
		return lines
	}
	start := match(file, 0, lines, string(args[1]))
	// File plus one arg: one line:
	//	!src file.go /foo/
	if len(args) == 2 {
		return [][]byte{lines[start]}
	}
	// File plus two args: range:
	//	!src file.go /foo/ /^}/
	end := match(file, start, lines, string(args[2]))
	return lines[start : end+1] // +1 to include matched line.
}

// htmlEscape makes sure input is HTML clean, if necessary.
func htmlEscape(input []byte) []byte {
	if !*html || bytes.IndexAny(input, `&"<>`) < 0 {
		return input
	}
	var b bytes.Buffer
	template.HTMLEscape(&b, input)
	return b.Bytes()
}

// readFile reads and returns a file as part of !src processing.
func readFile(name string) []byte {
	file, err := ioutil.ReadFile(name)
	if err != nil {
		log.Fatal(err)
	}
	return file
}

// match identifies the input line that matches the pattern in a !src invocation.
// If start>0, match lines starting there rather than at the beginning.
func match(file string, start int, lines [][]byte, pattern string) int {
	// $ matches the end of the file.
	if pattern == "$" {
		return len(lines) - 1
	}
	// Number matches the line.
	if i, err := strconv.Atoi(pattern); err == nil {
		return i - 1 // Lines are 1-indexed.
	}
	// /regexp/ matches the line that matches the regexp.
	if len(pattern) > 2 && pattern[0] == '/' && pattern[len(pattern)-1] == '/' {
		re, err := regexp.Compile(pattern[1 : len(pattern)-1])
		if err != nil {
			log.Fatal(err)
		}
		for i := start; i < len(lines); i++ {
			if re.Match(lines[i]) {
				return i
			}
		}
		log.Fatalf("%s: no match for %s", file, pattern)
	}
	log.Fatalf("unrecognized pattern: %s", pattern)
	return 0
}

// coalesce combines lines. Each time prefix is found on a line,
// it calls fold and replaces the line with return value from fold.
func coalesce(prefix []byte, fold func(i int) (n int, line []byte)) {
	j := 0 // output line number goes up by one each loop
	for i := 0; i < len(lines); {
		if bytes.HasPrefix(lines[i], prefix) {
			nlines, block := fold(i)
			lines[j] = block
			i += nlines
		} else {
			lines[j] = lines[i]
			i++
		}
		j++
	}
	lines = lines[0:j]
}

// foldPre returns the <pre> block as a single slice.
func foldPre(i int) (n int, line []byte) {
	buf := new(bytes.Buffer)
	for i < len(lines) {
		buf.Write(lines[i])
		n++
		if bytes.Equal(lines[i], preEnd) {
			break
		}
		i++
	}
	return n, buf.Bytes()
}

// foldTabs returns the tab-indented block as a single <pre>-bounded slice.
func foldTabs(i int) (n int, line []byte) {
	buf := new(bytes.Buffer)
	buf.WriteString("<pre>\n")
	for i < len(lines) {
		if !bytes.HasPrefix(lines[i], tab) {
			break
		}
		buf.Write(lines[i][1:]) // delete leading tab.
		n++
		i++
	}
	buf.WriteString("</pre>\n")
	return n, buf.Bytes()
}

// headings turns sections into HTML sections.
func headings() {
	b := bufio.NewWriter(os.Stdout)
	for i, l := range lines {
		if i > 0 && bytes.Equal(l, sectionMarker) {
			lines[i-1] = []byte("<h2>" + string(trim(lines[i-1])) + "</h2>\n")
			lines[i] = empty
		}
	}
	b.Flush()
}

// paragraphs turns blank lines into paragraph marks.
func paragraphs() {
	for i, l := range lines {
		if bytes.Equal(l, newline) {
			lines[i] = pp
		}
	}
}

// quotes turns "x" in the file into <code>x</code>.
func quotes() {
	for i, l := range lines {
		lines[i] = codeQuotes(l)
	}
}

// quotes turns "x" in the line into <code>x</code>.
func codeQuotes(l []byte) []byte {
	if bytes.HasPrefix(l, preStart) {
		return l
	}
	n := bytes.Index(l, quote)
	if n < 0 {
		return l
	}
	buf := new(bytes.Buffer)
	inQuote := false
	for _, c := range l {
		if c == '"' {
			if inQuote {
				buf.WriteString("</code>")
			} else {
				buf.WriteString("<code>")
			}
			inQuote = !inQuote
		} else {
			buf.WriteByte(c)
		}
	}
	return buf.Bytes()
}

// trim drops the trailing newline, if present.
func trim(l []byte) []byte {
	n := len(l)
	if n > 0 && l[n-1] == '\n' {
		return l[0 : n-1]
	}
	return l
}

// expandTabs expands tabs to spaces. It doesn't worry about columns.
func expandTabs(l []byte) []byte {
	return bytes.Replace(l, tab, indent, -1)
}
