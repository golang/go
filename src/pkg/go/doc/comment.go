// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Godoc comment extraction and comment -> HTML formatting.

package doc

import (
	"go/ast";
	"io";
	"strings";
	"template";	// for htmlEscape
)

// Comment extraction

// CommentText returns the text of comment,
// with the comment markers - //, /*, and */ - removed.
func CommentText(comment *ast.CommentGroup) string {
	if comment == nil {
		return "";
	}
	comments := make([]string, len(comment.List));
	for i, c := range comment.List {
		comments[i] = string(c.Text);
	}

	lines := make([]string, 0, 20);
	for _, c := range comments {
		// Remove comment markers.
		// The parser has given us exactly the comment text.
		switch n := len(c); {
		case n >= 4 && c[0:2] == "/*" && c[n-2 : n] == "*/":
			c = c[2 : n-2];
		case n >= 2 && c[0:2] == "//":
			c = c[2:n];
			// Remove leading space after //, if there is one.
			if len(c) > 0 && c[0] == ' ' {
				c = c[1:len(c)];
			}
		}

		// Split on newlines.
		cl := strings.Split(c, "\n", 0);

		// Walk lines, stripping trailing white space and adding to list.
		for _, l := range cl {
			// Strip trailing white space
			m := len(l);
			for m > 0 && (l[m-1] == ' ' || l[m-1] == '\n' || l[m-1] == '\t' || l[m-1] == '\r') {
				m--;
			}
			l = l[0:m];

			// Add to list.
			n := len(lines);
			if n+1 >= cap(lines) {
				newlines := make([]string, n, 2*cap(lines));
				for k := range newlines {
					newlines[k] = lines[k];
				}
				lines = newlines;
			}
			lines = lines[0 : n+1];
			lines[n] = l;
		}
	}

	// Remove leading blank lines; convert runs of
	// interior blank lines to a single blank line.
	n := 0;
	for _, line := range lines {
		if line != "" || n > 0 && lines[n-1] != "" {
			lines[n] = line;
			n++;
		}
	}
	lines = lines[0:n];

	// Add final "" entry to get trailing newline from Join.
	// The original loop always leaves room for one more.
	if n > 0 && lines[n-1] != "" {
		lines = lines[0 : n+1];
		lines[n] = "";
	}

	return strings.Join(lines, "\n");
}

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
		out[n] = text[last:len(text)];
	}

	return out;
}


var (
	ldquo	= strings.Bytes("&ldquo;");
	rdquo	= strings.Bytes("&rdquo;");
)

// Escape comment text for HTML.
// Also, turn `` into &ldquo; and '' into &rdquo;.
func commentEscape(w io.Writer, s []byte) {
	last := 0;
	for i := 0; i < len(s)-1; i++ {
		if s[i] == s[i+1] && (s[i] == '`' || s[i] == '\'') {
			template.HtmlEscape(w, s[last:i]);
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
	template.HtmlEscape(w, s[last:len(s)]);
}


var (
	html_p		= strings.Bytes("<p>\n");
	html_endp	= strings.Bytes("</p>\n");
	html_pre	= strings.Bytes("<pre>");
	html_endpre	= strings.Bytes("</pre>\n");
)


func indentLen(s []byte) int {
	i := 0;
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++;
	}
	return i;
}


func isBlank(s []byte) bool {
	return len(s) == 0 || (len(s) == 1 && s[0] == '\n');
}


func commonPrefix(a, b []byte) []byte {
	i := 0;
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++;
	}
	return a[0:i];
}


func unindent(block [][]byte) {
	if len(block) == 0 {
		return;
	}

	// compute maximum common white prefix
	prefix := block[0][0 : indentLen(block[0])];
	for _, line := range block {
		if !isBlank(line) {
			prefix = commonPrefix(prefix, line[0 : indentLen(line)]);
		}
	}
	n := len(prefix);

	// remove
	for i, line := range block {
		if !isBlank(line) {
			block[i] = line[n:len(line)];
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
func ToHtml(w io.Writer, s []byte) {
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
	for i := 0; i < len(lines); {
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
			block := lines[i:j];
			i = j;

			unindent(block);

			// put those lines in a pre block.
			// they don't get the nice text formatting,
			// just html escaping
			w.Write(html_pre);
			for _, line := range block {
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
