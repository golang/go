// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Process plain text into HTML.
//	- h2's are made from lines followed by a line "----\n"
//	- tab-indented blocks become <pre> blocks
//	- blank lines become <p> marks
//	- "quoted strings" become <code>quoted strings</code>

package main

import (
	"bufio";
	"bytes";
	"log";
	"os";
	"strings";
)

var (
	lines = make([][]byte, 0, 10000);	// assume big enough
	linebuf = make([]byte, 10000);		// assume big enough

	empty = strings.Bytes("");
	newline = strings.Bytes("\n");
	tab = strings.Bytes("\t");
	quote = strings.Bytes(`"`);

	sectionMarker = strings.Bytes("----\n");
	preStart = strings.Bytes("<pre>");
	preEnd = strings.Bytes("</pre>\n");
	pp = strings.Bytes("<p>\n");
);

func main() {
	read();
	headings();
	paragraphs();
	coalesce(preStart, foldPre);
	coalesce(tab, foldTabs);
	quotes();
	write();
}

func read() {
	b := bufio.NewReader(os.Stdin);
	for {
		line, err := b.ReadBytes('\n');
		if err == os.EOF {
			break;
		}
		if err != nil {
			log.Exit(err)
		}
		n := len(lines);
		lines = lines[0:n+1];
		lines[n] = line;
	}
}

func write() {
	b := bufio.NewWriter(os.Stdout);
	for _, line := range lines {
		b.Write(expandTabs(line));
	}
	b.Flush();
}

// each time prefix is found on a line, call fold and replace
// line with return value from fold.
func coalesce(prefix []byte, fold func(i int) (n int, line []byte)) {
	j := 0;	// output line number; goes up by one each loop
	for i := 0; i < len(lines); {
		if bytes.HasPrefix(lines[i], prefix) {
			nlines, block := fold(i);
			lines[j] = block;
			i += nlines;
		} else {
			lines[j] = lines[i];
			i++;
		}
		j++;
	}
	lines = lines[0:j];
}

// return the <pre> block as a single slice
func foldPre(i int) (n int, line []byte) {
	buf := new(bytes.Buffer);
	for i < len(lines) {
		buf.Write(lines[i]);
		n++;
		if bytes.Equal(lines[i], preEnd) {
			break
		}
		i++;
	}
	return n, buf.Bytes();
}

// return the tab-indented block as a single <pre>-bounded slice
func foldTabs(i int) (n int, line []byte) {
	buf := new(bytes.Buffer);
	buf.WriteString("<pre>\n");
	for i < len(lines) {
		if !bytes.HasPrefix(lines[i], tab) {
			break;
		}
		buf.Write(lines[i]);
		n++;
		i++;
	}
	buf.WriteString("</pre>\n");
	return n, buf.Bytes();
}

func headings() {
	b := bufio.NewWriter(os.Stdout);
	for i, l := range lines {
		if i > 0 && bytes.Equal(l, sectionMarker) {
			lines[i-1] = strings.Bytes("<h2>" + string(trim(lines[i-1])) + "</h2>\n");
			lines[i] = empty;
		}
	}
	b.Flush();
}

func paragraphs() {
	for i, l := range lines {
		if bytes.Equal(l, newline) {
			lines[i] = pp;
		}
	}
}

func quotes() {
	for i, l := range lines {
		lines[i] = codeQuotes(l);
	}
}

func codeQuotes(l []byte) []byte {
	if bytes.HasPrefix(l, preStart) {
		return l
	}
	n := bytes.Index(l, quote);
	if n < 0 {
		return l
	}
	buf := new(bytes.Buffer);
	inQuote := false;
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
	return buf.Bytes();
}

// drop trailing newline
func trim(l []byte) []byte {
	n := len(l);
	if n > 0 && l[n-1] == '\n' {
		return l[0:n-1]
	}
	return l
}

// expand tabs to 4 spaces. don't worry about columns.
func expandTabs(l []byte) []byte {
	j := 0;	// position in linebuf.
	for _, c := range l {
		if c == '\t' {
			for k := 0; k < 4; k++ {
				linebuf[j] = ' ';
				j++;
			}
		} else {
			linebuf[j] = c;
			j++;
		}
	}
	return linebuf[0:j];
}
