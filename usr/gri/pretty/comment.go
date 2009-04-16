// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Godoc comment -> HTML formatting

package comment

import (
	"fmt";
	"io";
	"template";
)

// Split bytes into lines.
func split(text []byte) [][]byte {
	// count lines
	n := 0;
	last := 0;
	for i, c := range text {
		if c == '\n' {
			last = i+1;
			n++;
		}
	}
	if last < len(text) {
		n++;
	}

	// split
	out := make([][]byte, n);
	last = 0;
	n = 0;
	for i, c := range text {
		if c == '\n' {
			out[n] = text[last : i+1];
			last = i+1;
			n++;
		}
	}
	if last < len(text) {
		out[n] = text[last : len(text)];
	}

	return out;
}


var (
	ldquo = io.StringBytes("&ldquo;");
	rdquo = io.StringBytes("&rdquo;");
)

// Escape comment text for HTML.
// Also, turn `` into &ldquo; and '' into &rdquo;.
func commentEscape(w io.Write, s []byte) {
	last := 0;
	for i := 0; i < len(s)-1; i++ {
		if s[i] == s[i+1] && (s[i] == '`' || s[i] == '\'') {
			template.HtmlEscape(w, s[last : i]);
			last = i+2;
			switch s[i] {
			case '`':
				w.Write(ldquo);
			case '\'':
				w.Write(rdquo);
			}
			i++;	// loop will add one more
		}
	}
	template.HtmlEscape(w, s[last : len(s)]);
}


var (
	html_p = io.StringBytes("<p>\n");
	html_endp = io.StringBytes("</p>\n");
	html_pre = io.StringBytes("<pre>");
	html_endpre = io.StringBytes("</pre>\n");
)


func indentLen(s []byte) int {
	i := 0;
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++;
	}
	return i;
}


func isBlank(s []byte) bool {
	return len(s) == 0 || (len(s) == 1 && s[0] == '\n')
}


func commonPrefix(a, b []byte) []byte {
	i := 0;
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++;
	}
	return a[0 : i];
}


func unindent(block [][]byte) {
	if len(block) == 0 {
		return;
	}

	// compute maximum common white prefix
	prefix := block[0][0 : indentLen(block[0])];
	for i, line := range block {
		if !isBlank(line) {
			prefix = commonPrefix(prefix, line[0 : indentLen(line)]);
		}
	}
	n := len(prefix);

	// remove
	for i, line := range block {
		if !isBlank(line) {
			block[i] = line[n : len(line)];
		}
	}
}


// Convert comment text to formatted HTML.
// The comment was prepared by DocReader,
// so it is known not to have leading, trailing blank lines
// nor to have trailing spaces at the end of lines.
// The comment markers have already been removed.
//
// Turn each run of multiple \n into </p><p>
// Turn each run of indented lines into <pre> without indent.
//
// TODO(rsc): I'd like to pass in an array of variable names []string
// and then italicize those strings when they appear as words.
func ToHtml(w io.Write, s []byte) {
	inpara := false;

	/* TODO(rsc): 6g cant generate code for these
	close := func() {
		if inpara {
			w.Write(html_endp);
			inpara = false;
		}
	};
	open := func() {
		if !inpara {
			w.Write(html_p);
			inpara = true;
		}
	};
	*/

	lines := split(s);
	unindent(lines);
	for i := 0; i < len(lines);  {
		line := lines[i];
		if isBlank(line) {
			// close paragraph
			if inpara {
				w.Write(html_endp);
				inpara = false;
			}
			i++;
			continue;
		}
		if indentLen(line) > 0 {
			// close paragraph
			if inpara {
				w.Write(html_endp);
				inpara = false;
			}

			// count indented or blank lines
			j := i+1;
			for j < len(lines) && (isBlank(lines[j]) || indentLen(lines[j]) > 0) {
				j++;
			}
			// but not trailing blank lines
			for j > i && isBlank(lines[j-1]) {
				j--;
			}
			block := lines[i : j];
			i = j;

			unindent(block);

			// put those lines in a pre block.
			// they don't get the nice text formatting,
			// just html escaping
			w.Write(html_pre);
			for k, line := range block {
				template.HtmlEscape(w, line);
			}
			w.Write(html_endpre);
			continue;
		}
		// open paragraph
		if !inpara {
			w.Write(html_p);
			inpara = true;
		}
		commentEscape(w, lines[i]);
		i++;
	}
	if inpara {
		w.Write(html_endp);
		inpara = false;
	}
}

