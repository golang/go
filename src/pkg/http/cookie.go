// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// A note on Version=0 vs. Version=1 cookies
//
// The difference between Set-Cookie and Set-Cookie2 is hard to discern from the
// RFCs as it is not stated explicitly.  There seem to be three standards
// lingering on the web: Netscape, RFC 2109 (aka Version=0) and RFC 2965 (aka
// Version=1). It seems that Netscape and RFC 2109 are the same thing, hereafter
// Version=0 cookies.
//
// In general, Set-Cookie2 is a superset of Set-Cookie. It has a few new
// attributes like HttpOnly and Secure.  To be meticulous, if a server intends
// to use these, it needs to send a Set-Cookie2.  However, it is most likely
// most modern browsers will not complain seeing an HttpOnly attribute in a
// Set-Cookie header.
//
// Both RFC 2109 and RFC 2965 use Cookie in the same way - two send cookie
// values from clients to servers - and the allowable attributes seem to be the
// same.
// 
// The Cookie2 header is used for a different purpose. If a client suspects that
// the server speaks Version=1 (RFC 2965) then along with the Cookie header
// lines, you can also send:
//
//   Cookie2: $Version="1"
//
// in order to suggest to the server that you understand Version=1 cookies. At
// which point the server may continue responding with Set-Cookie2 headers.
// When a client sends the (above) Cookie2 header line, it must be prepated to
// understand incoming Set-Cookie2.
//
// This implementation of cookies supports neither Set-Cookie2 nor Cookie2
// headers. However, it parses Version=1 Cookies (along with Version=0) as well
// as Set-Cookie headers which utilize the full Set-Cookie2 syntax.

// TODO(petar): Explicitly forbid parsing of Set-Cookie attributes
// starting with '$', which have been used to hack into broken
// servers using the eventual Request headers containing those
// invalid attributes that may overwrite intended $Version, $Path, 
// etc. attributes.
// TODO(petar): Read 'Set-Cookie2' headers and prioritize them over equivalent
// 'Set-Cookie' headers. 'Set-Cookie2' headers are still extremely rare.

// A Cookie represents an RFC 2965 HTTP cookie as sent in
// the Set-Cookie header of an HTTP response or the Cookie header
// of an HTTP request.
// The Set-Cookie2 and Cookie2 headers are unimplemented.
type Cookie struct {
	Name       string
	Value      string
	Path       string
	Domain     string
	Comment    string
	Version    int
	Expires    time.Time
	RawExpires string
	MaxAge     int // Max age in seconds
	Secure     bool
	HttpOnly   bool
	Raw        string
	Unparsed   []string // Raw text of unparsed attribute-value pairs
}

// readSetCookies parses all "Set-Cookie" values from
// the header h, removes the successfully parsed values from the 
// "Set-Cookie" key in h and returns the parsed Cookies.
func readSetCookies(h Header) []*Cookie {
	cookies := []*Cookie{}
	var unparsedLines []string
	for _, line := range h["Set-Cookie"] {
		parts := strings.Split(strings.TrimSpace(line), ";", -1)
		if len(parts) == 1 && parts[0] == "" {
			continue
		}
		parts[0] = strings.TrimSpace(parts[0])
		j := strings.Index(parts[0], "=")
		if j < 0 {
			unparsedLines = append(unparsedLines, line)
			continue
		}
		name, value := parts[0][:j], parts[0][j+1:]
		value, err := URLUnescape(value)
		if err != nil {
			unparsedLines = append(unparsedLines, line)
			continue
		}
		c := &Cookie{
			Name:   name,
			Value:  value,
			MaxAge: -1, // Not specified
			Raw:    line,
		}
		for i := 1; i < len(parts); i++ {
			parts[i] = strings.TrimSpace(parts[i])
			if len(parts[i]) == 0 {
				continue
			}

			attr, val := parts[i], ""
			if j := strings.Index(attr, "="); j >= 0 {
				attr, val = attr[:j], attr[j+1:]
				val, err = URLUnescape(val)
				if err != nil {
					c.Unparsed = append(c.Unparsed, parts[i])
					continue
				}
			}
			switch strings.ToLower(attr) {
			case "secure":
				c.Secure = true
				continue
			case "httponly":
				c.HttpOnly = true
				continue
			case "comment":
				c.Comment = val
				continue
			case "domain":
				c.Domain = val
				// TODO: Add domain parsing
				continue
			case "max-age":
				secs, err := strconv.Atoi(val)
				if err != nil || secs < 0 {
					break
				}
				c.MaxAge = secs
				continue
			case "expires":
				c.RawExpires = val
				exptime, err := time.Parse(time.RFC1123, val)
				if err != nil {
					c.Expires = time.Time{}
					break
				}
				c.Expires = *exptime
				continue
			case "path":
				c.Path = val
				// TODO: Add path parsing
				continue
			case "version":
				c.Version, err = strconv.Atoi(val)
				if err != nil {
					c.Version = 0
					break
				}
				continue
			}
			c.Unparsed = append(c.Unparsed, parts[i])
		}
		cookies = append(cookies, c)
	}
	h["Set-Cookie"] = unparsedLines, unparsedLines != nil
	return cookies
}

// writeSetCookies writes the wire representation of the set-cookies
// to w. Each cookie is written on a separate "Set-Cookie: " line.
// This choice is made because HTTP parsers tend to have a limit on
// line-length, so it seems safer to place cookies on separate lines.
func writeSetCookies(w io.Writer, kk []*Cookie) os.Error {
	if kk == nil {
		return nil
	}
	lines := make([]string, 0, len(kk))
	var b bytes.Buffer
	for _, c := range kk {
		b.Reset()
		// TODO(petar): c.Value (below) should be unquoted if it is recognized as quoted
		fmt.Fprintf(&b, "%s=%s", CanonicalHeaderKey(c.Name), c.Value)
		if c.Version > 0 {
			fmt.Fprintf(&b, "Version=%d; ", c.Version)
		}
		if len(c.Path) > 0 {
			fmt.Fprintf(&b, "; Path=%s", URLEscape(c.Path))
		}
		if len(c.Domain) > 0 {
			fmt.Fprintf(&b, "; Domain=%s", URLEscape(c.Domain))
		}
		if len(c.Expires.Zone) > 0 {
			fmt.Fprintf(&b, "; Expires=%s", c.Expires.Format(time.RFC1123))
		}
		if c.MaxAge >= 0 {
			fmt.Fprintf(&b, "; Max-Age=%d", c.MaxAge)
		}
		if c.HttpOnly {
			fmt.Fprintf(&b, "; HttpOnly")
		}
		if c.Secure {
			fmt.Fprintf(&b, "; Secure")
		}
		if len(c.Comment) > 0 {
			fmt.Fprintf(&b, "; Comment=%s", URLEscape(c.Comment))
		}
		lines = append(lines, "Set-Cookie: "+b.String()+"\r\n")
	}
	sort.SortStrings(lines)
	for _, l := range lines {
		if _, err := io.WriteString(w, l); err != nil {
			return err
		}
	}
	return nil
}

// readCookies parses all "Cookie" values from
// the header h, removes the successfully parsed values from the 
// "Cookie" key in h and returns the parsed Cookies.
func readCookies(h Header) []*Cookie {
	cookies := []*Cookie{}
	lines, ok := h["Cookie"]
	if !ok {
		return cookies
	}
	unparsedLines := []string{}
	for _, line := range lines {
		parts := strings.Split(strings.TrimSpace(line), ";", -1)
		if len(parts) == 1 && parts[0] == "" {
			continue
		}
		// Per-line attributes
		var lineCookies = make(map[string]string)
		var version int
		var path string
		var domain string
		var comment string
		var httponly bool
		for i := 0; i < len(parts); i++ {
			parts[i] = strings.TrimSpace(parts[i])
			if len(parts[i]) == 0 {
				continue
			}
			attr, val := parts[i], ""
			var err os.Error
			if j := strings.Index(attr, "="); j >= 0 {
				attr, val = attr[:j], attr[j+1:]
				val, err = URLUnescape(val)
				if err != nil {
					continue
				}
			}
			switch strings.ToLower(attr) {
			case "$httponly":
				httponly = true
			case "$version":
				version, err = strconv.Atoi(val)
				if err != nil {
					version = 0
					continue
				}
			case "$domain":
				domain = val
				// TODO: Add domain parsing
			case "$path":
				path = val
				// TODO: Add path parsing
			case "$comment":
				comment = val
			default:
				lineCookies[attr] = val
			}
		}
		if len(lineCookies) == 0 {
			unparsedLines = append(unparsedLines, line)
		}
		for n, v := range lineCookies {
			cookies = append(cookies, &Cookie{
				Name:     n,
				Value:    v,
				Path:     path,
				Domain:   domain,
				Comment:  comment,
				Version:  version,
				HttpOnly: httponly,
				MaxAge:   -1,
				Raw:      line,
			})
		}
	}
	h["Cookie"] = unparsedLines, len(unparsedLines) > 0
	return cookies
}

// writeCookies writes the wire representation of the cookies
// to w. Each cookie is written on a separate "Cookie: " line.
// This choice is made because HTTP parsers tend to have a limit on
// line-length, so it seems safer to place cookies on separate lines.
func writeCookies(w io.Writer, kk []*Cookie) os.Error {
	lines := make([]string, 0, len(kk))
	var b bytes.Buffer
	for _, c := range kk {
		b.Reset()
		n := c.Name
		if c.Version > 0 {
			fmt.Fprintf(&b, "$Version=%d; ", c.Version)
		}
		// TODO(petar): c.Value (below) should be unquoted if it is recognized as quoted
		fmt.Fprintf(&b, "%s=%s", CanonicalHeaderKey(n), c.Value)
		if len(c.Path) > 0 {
			fmt.Fprintf(&b, "; $Path=%s", URLEscape(c.Path))
		}
		if len(c.Domain) > 0 {
			fmt.Fprintf(&b, "; $Domain=%s", URLEscape(c.Domain))
		}
		if c.HttpOnly {
			fmt.Fprintf(&b, "; $HttpOnly")
		}
		if len(c.Comment) > 0 {
			fmt.Fprintf(&b, "; $Comment=%s", URLEscape(c.Comment))
		}
		lines = append(lines, "Cookie: "+b.String()+"\r\n")
	}
	sort.SortStrings(lines)
	for _, l := range lines {
		if _, err := io.WriteString(w, l); err != nil {
			return err
		}
	}
	return nil
}
