// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import (
	"fmt"
	"log"
	"net/url"
	"strings"
)

func init() {
	Register("link", parseLink)
}

type Link struct {
	URL   *url.URL
	Label string
}

func (l Link) TemplateName() string { return "link" }

func parseLink(ctx *Context, fileName string, lineno int, text string) (Elem, error) {
	args := strings.Fields(text)
	url, err := url.Parse(args[1])
	if err != nil {
		return nil, err
	}
	label := ""
	if len(args) > 2 {
		label = strings.Join(args[2:], " ")
	} else {
		scheme := url.Scheme + "://"
		if url.Scheme == "mailto" {
			scheme = "mailto:"
		}
		label = strings.Replace(url.String(), scheme, "", 1)
	}
	return Link{url, label}, nil
}

func renderLink(href, text string) string {
	text = font(text)
	if text == "" {
		text = href
	}
	// Open links in new window only when their url is absolute.
	target := "_blank"
	if u, err := url.Parse(href); err != nil {
		log.Println("rendernLink parsing url:", err)
	} else if !u.IsAbs() || u.Scheme == "javascript" {
		target = "_self"
	}

	return fmt.Sprintf(`<a href="%s" target="%s">%s</a>`, href, target, text)
}

// parseInlineLink parses an inline link at the start of s, and returns
// a rendered HTML link and the total length of the raw inline link.
// If no inline link is present, it returns all zeroes.
func parseInlineLink(s string) (link string, length int) {
	if !strings.HasPrefix(s, "[[") {
		return
	}
	end := strings.Index(s, "]]")
	if end == -1 {
		return
	}
	urlEnd := strings.Index(s, "]")
	rawURL := s[2:urlEnd]
	const badURLChars = `<>"{}|\^[] ` + "`" // per RFC2396 section 2.4.3
	if strings.ContainsAny(rawURL, badURLChars) {
		return
	}
	if urlEnd == end {
		simpleUrl := ""
		url, err := url.Parse(rawURL)
		if err == nil {
			// If the URL is http://foo.com, drop the http://
			// In other words, render [[http://golang.org]] as:
			//   <a href="http://golang.org">golang.org</a>
			if strings.HasPrefix(rawURL, url.Scheme+"://") {
				simpleUrl = strings.TrimPrefix(rawURL, url.Scheme+"://")
			} else if strings.HasPrefix(rawURL, url.Scheme+":") {
				simpleUrl = strings.TrimPrefix(rawURL, url.Scheme+":")
			}
		}
		return renderLink(rawURL, simpleUrl), end + 2
	}
	if s[urlEnd:urlEnd+2] != "][" {
		return
	}
	text := s[urlEnd+2 : end]
	return renderLink(rawURL, text), end + 2
}
