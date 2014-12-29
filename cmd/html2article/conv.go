// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program takes an HTML file and outputs a corresponding article file in
// present format. See: golang.org/x/tools/present
package main // import "golang.org/x/tools/cmd/html2article"

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/url"
	"os"
	"regexp"
	"strings"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

func main() {
	flag.Parse()

	err := convert(os.Stdout, os.Stdin)
	if err != nil {
		log.Fatal(err)
	}
}

func convert(w io.Writer, r io.Reader) error {
	root, err := html.Parse(r)
	if err != nil {
		return err
	}

	style := find(root, isTag(atom.Style))
	parseStyles(style)

	body := find(root, isTag(atom.Body))
	if body == nil {
		return errors.New("couldn't find body")
	}
	article := limitNewlineRuns(makeHeadings(strings.TrimSpace(text(body))))
	_, err = fmt.Fprintf(w, "Title\n\n%s", article)
	return err
}

type Style string

const (
	Bold   Style = "*"
	Italic Style = "_"
	Code   Style = "`"
)

var cssRules = make(map[string]Style)

func parseStyles(style *html.Node) {
	if style == nil || style.FirstChild == nil {
		log.Println("couldn't find styles")
		return
	}
	s := bufio.NewScanner(strings.NewReader(style.FirstChild.Data))

	findRule := func(b []byte, atEOF bool) (advance int, token []byte, err error) {
		if i := bytes.Index(b, []byte("{")); i >= 0 {
			token = bytes.TrimSpace(b[:i])
			advance = i
		}
		return
	}
	findBody := func(b []byte, atEOF bool) (advance int, token []byte, err error) {
		if len(b) == 0 {
			return
		}
		if b[0] != '{' {
			err = fmt.Errorf("expected {, got %c", b[0])
			return
		}
		if i := bytes.Index(b, []byte("}")); i < 0 {
			err = fmt.Errorf("can't find closing }")
			return
		} else {
			token = b[1:i]
			advance = i + 1
		}
		return
	}

	s.Split(findRule)
	for s.Scan() {
		rule := s.Text()
		s.Split(findBody)
		if !s.Scan() {
			break
		}
		b := strings.ToLower(s.Text())
		switch {
		case strings.Contains(b, "italic"):
			cssRules[rule] = Italic
		case strings.Contains(b, "bold"):
			cssRules[rule] = Bold
		case strings.Contains(b, "Consolas") || strings.Contains(b, "Courier New"):
			cssRules[rule] = Code
		}
		s.Split(findRule)
	}
	if err := s.Err(); err != nil {
		log.Println(err)
	}
}

var newlineRun = regexp.MustCompile(`\n\n+`)

func limitNewlineRuns(s string) string {
	return newlineRun.ReplaceAllString(s, "\n\n")
}

func makeHeadings(body string) string {
	buf := new(bytes.Buffer)
	lines := strings.Split(body, "\n")
	for i, s := range lines {
		if i == 0 && !isBoldTitle(s) {
			buf.WriteString("* Introduction\n\n")
		}
		if isBoldTitle(s) {
			s = strings.TrimSpace(strings.Replace(s, "*", " ", -1))
			s = "* " + s
		}
		buf.WriteString(s)
		buf.WriteByte('\n')
	}
	return buf.String()
}

func isBoldTitle(s string) bool {
	return !strings.Contains(s, " ") &&
		strings.HasPrefix(s, "*") &&
		strings.HasSuffix(s, "*")
}

func indent(buf *bytes.Buffer, s string) {
	for _, l := range strings.Split(s, "\n") {
		if l != "" {
			buf.WriteByte('\t')
			buf.WriteString(l)
		}
		buf.WriteByte('\n')
	}
}

func unwrap(buf *bytes.Buffer, s string) {
	var cont bool
	for _, l := range strings.Split(s, "\n") {
		l = strings.TrimSpace(l)
		if len(l) == 0 {
			if cont {
				buf.WriteByte('\n')
				buf.WriteByte('\n')
			}
			cont = false
		} else {
			if cont {
				buf.WriteByte(' ')
			}
			buf.WriteString(l)
			cont = true
		}
	}
}

func text(n *html.Node) string {
	var buf bytes.Buffer
	walk(n, func(n *html.Node) bool {
		switch n.Type {
		case html.TextNode:
			buf.WriteString(n.Data)
			return false
		case html.ElementNode:
			// no-op
		default:
			return true
		}
		a := n.DataAtom
		if a == atom.Span {
			switch {
			case hasStyle(Code)(n):
				a = atom.Code
			case hasStyle(Bold)(n):
				a = atom.B
			case hasStyle(Italic)(n):
				a = atom.I
			}
		}
		switch a {
		case atom.Br:
			buf.WriteByte('\n')
		case atom.P:
			unwrap(&buf, childText(n))
			buf.WriteString("\n\n")
		case atom.Li:
			buf.WriteString("- ")
			unwrap(&buf, childText(n))
			buf.WriteByte('\n')
		case atom.Pre:
			indent(&buf, childText(n))
			buf.WriteByte('\n')
		case atom.A:
			href, text := attr(n, "href"), childText(n)
			// Skip links with no text.
			if strings.TrimSpace(text) == "" {
				break
			}
			// Don't emit empty links.
			if strings.TrimSpace(href) == "" {
				buf.WriteString(text)
				break
			}
			// Use original url for Google Docs redirections.
			if u, err := url.Parse(href); err != nil {
				log.Printf("parsing url %q: %v", href, err)
			} else if u.Host == "www.google.com" && u.Path == "/url" {
				href = u.Query().Get("q")
			}
			fmt.Fprintf(&buf, "[[%s][%s]]", href, text)
		case atom.Code:
			buf.WriteString(highlight(n, "`"))
		case atom.B:
			buf.WriteString(highlight(n, "*"))
		case atom.I:
			buf.WriteString(highlight(n, "_"))
		case atom.Img:
			src := attr(n, "src")
			fmt.Fprintf(&buf, ".image %s\n", src)
		case atom.Iframe:
			src, w, h := attr(n, "src"), attr(n, "width"), attr(n, "height")
			fmt.Fprintf(&buf, "\n.iframe %s %s %s\n", src, h, w)
		case atom.Param:
			if attr(n, "name") == "movie" {
				// Old style YouTube embed.
				u := attr(n, "value")
				u = strings.Replace(u, "/v/", "/embed/", 1)
				if i := strings.Index(u, "&"); i >= 0 {
					u = u[:i]
				}
				fmt.Fprintf(&buf, "\n.iframe %s 540 304\n", u)
			}
		case atom.Title:
		default:
			return true
		}
		return false
	})
	return buf.String()
}

func childText(node *html.Node) string {
	var buf bytes.Buffer
	for n := node.FirstChild; n != nil; n = n.NextSibling {
		fmt.Fprint(&buf, text(n))
	}
	return buf.String()
}

func highlight(node *html.Node, char string) string {
	t := strings.Replace(childText(node), " ", char, -1)
	return fmt.Sprintf("%s%s%s", char, t, char)
}

type selector func(*html.Node) bool

func isTag(a atom.Atom) selector {
	return func(n *html.Node) bool {
		return n.DataAtom == a
	}
}

func hasClass(name string) selector {
	return func(n *html.Node) bool {
		for _, a := range n.Attr {
			if a.Key == "class" {
				for _, c := range strings.Fields(a.Val) {
					if c == name {
						return true
					}
				}
			}
		}
		return false
	}
}

func hasStyle(s Style) selector {
	return func(n *html.Node) bool {
		for rule, s2 := range cssRules {
			if s2 != s {
				continue
			}
			if strings.HasPrefix(rule, ".") && hasClass(rule[1:])(n) {
				return true
			}
			if n.DataAtom.String() == rule {
				return true
			}
		}
		return false
	}
}

func hasAttr(key, val string) selector {
	return func(n *html.Node) bool {
		for _, a := range n.Attr {
			if a.Key == key && a.Val == val {
				return true
			}
		}
		return false
	}
}

func attr(node *html.Node, key string) (value string) {
	for _, attr := range node.Attr {
		if attr.Key == key {
			return attr.Val
		}
	}
	return ""
}

func findAll(node *html.Node, fn selector) (nodes []*html.Node) {
	walk(node, func(n *html.Node) bool {
		if fn(n) {
			nodes = append(nodes, n)
		}
		return true
	})
	return
}

func find(n *html.Node, fn selector) *html.Node {
	var result *html.Node
	walk(n, func(n *html.Node) bool {
		if result != nil {
			return false
		}
		if fn(n) {
			result = n
			return false
		}
		return true
	})
	return result
}

func walk(n *html.Node, fn selector) {
	if fn(n) {
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c, fn)
		}
	}
}
