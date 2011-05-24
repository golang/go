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

// This implementation is done according to RFC 6265:
//
//    http://tools.ietf.org/html/rfc6265

// A Cookie represents an HTTP cookie as sent in the Set-Cookie header of an
// HTTP response or the Cookie header of an HTTP request.
type Cookie struct {
	Name       string
	Value      string
	Path       string
	Domain     string
	Expires    time.Time
	RawExpires string

	// MaxAge=0 means no 'Max-Age' attribute specified. 
	// MaxAge<0 means delete cookie now, equivalently 'Max-Age: 0'
	// MaxAge>0 means Max-Age attribute present and given in seconds
	MaxAge   int
	Secure   bool
	HttpOnly bool
	Raw      string
	Unparsed []string // Raw text of unparsed attribute-value pairs
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
		if !isCookieNameValid(name) {
			unparsedLines = append(unparsedLines, line)
			continue
		}
		value, success := parseCookieValue(value)
		if !success {
			unparsedLines = append(unparsedLines, line)
			continue
		}
		c := &Cookie{
			Name:  name,
			Value: value,
			Raw:   line,
		}
		for i := 1; i < len(parts); i++ {
			parts[i] = strings.TrimSpace(parts[i])
			if len(parts[i]) == 0 {
				continue
			}

			attr, val := parts[i], ""
			if j := strings.Index(attr, "="); j >= 0 {
				attr, val = attr[:j], attr[j+1:]
			}
			lowerAttr := strings.ToLower(attr)
			parseCookieValueFn := parseCookieValue
			if lowerAttr == "expires" {
				parseCookieValueFn = parseCookieExpiresValue
			}
			val, success = parseCookieValueFn(val)
			if !success {
				c.Unparsed = append(c.Unparsed, parts[i])
				continue
			}
			switch lowerAttr {
			case "secure":
				c.Secure = true
				continue
			case "httponly":
				c.HttpOnly = true
				continue
			case "domain":
				c.Domain = val
				// TODO: Add domain parsing
				continue
			case "max-age":
				secs, err := strconv.Atoi(val)
				if err != nil || secs < 0 || secs != 0 && val[0] == '0' {
					break
				}
				if secs <= 0 {
					c.MaxAge = -1
				} else {
					c.MaxAge = secs
				}
				continue
			case "expires":
				c.RawExpires = val
				exptime, err := time.Parse(time.RFC1123, val)
				if err != nil {
					exptime, err = time.Parse("Mon, 02-Jan-2006 15:04:05 MST", val)
					if err != nil {
						c.Expires = time.Time{}
						break
					}
				}
				c.Expires = *exptime
				continue
			case "path":
				c.Path = val
				// TODO: Add path parsing
				continue
			}
			c.Unparsed = append(c.Unparsed, parts[i])
		}
		cookies = append(cookies, c)
	}
	h["Set-Cookie"] = unparsedLines, unparsedLines != nil
	return cookies
}

// SetCookie adds a Set-Cookie header to the provided ResponseWriter's headers.
func SetCookie(w ResponseWriter, cookie *Cookie) {
	var b bytes.Buffer
	writeSetCookieToBuffer(&b, cookie)
	w.Header().Add("Set-Cookie", b.String())
}

func writeSetCookieToBuffer(buf *bytes.Buffer, c *Cookie) {
	fmt.Fprintf(buf, "%s=%s", sanitizeName(c.Name), sanitizeValue(c.Value))
	if len(c.Path) > 0 {
		fmt.Fprintf(buf, "; Path=%s", sanitizeValue(c.Path))
	}
	if len(c.Domain) > 0 {
		fmt.Fprintf(buf, "; Domain=%s", sanitizeValue(c.Domain))
	}
	if len(c.Expires.Zone) > 0 {
		fmt.Fprintf(buf, "; Expires=%s", c.Expires.Format(time.RFC1123))
	}
	if c.MaxAge > 0 {
		fmt.Fprintf(buf, "; Max-Age=%d", c.MaxAge)
	} else if c.MaxAge < 0 {
		fmt.Fprintf(buf, "; Max-Age=0")
	}
	if c.HttpOnly {
		fmt.Fprintf(buf, "; HttpOnly")
	}
	if c.Secure {
		fmt.Fprintf(buf, "; Secure")
	}
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
		writeSetCookieToBuffer(&b, c)
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
		parsedPairs := 0
		for i := 0; i < len(parts); i++ {
			parts[i] = strings.TrimSpace(parts[i])
			if len(parts[i]) == 0 {
				continue
			}
			attr, val := parts[i], ""
			if j := strings.Index(attr, "="); j >= 0 {
				attr, val = attr[:j], attr[j+1:]
			}
			if !isCookieNameValid(attr) {
				continue
			}
			val, success := parseCookieValue(val)
			if !success {
				continue
			}
			cookies = append(cookies, &Cookie{Name: attr, Value: val})
			parsedPairs++
		}
		if parsedPairs == 0 {
			unparsedLines = append(unparsedLines, line)
		}
	}
	h["Cookie"] = unparsedLines, len(unparsedLines) > 0
	return cookies
}

// writeCookies writes the wire representation of the cookies to
// w. According to RFC 6265 section 5.4, writeCookies does not
// attach more than one Cookie header field.  That means all
// cookies, if any, are written into the same line, separated by
// semicolon.
func writeCookies(w io.Writer, kk []*Cookie) os.Error {
	if len(kk) == 0 {
		return nil
	}
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Cookie: ")
	for i, c := range kk {
		if i > 0 {
			fmt.Fprintf(&buf, "; ")
		}
		fmt.Fprintf(&buf, "%s=%s", sanitizeName(c.Name), sanitizeValue(c.Value))
	}
	fmt.Fprintf(&buf, "\r\n")
	_, err := w.Write(buf.Bytes())
	return err
}

func sanitizeName(n string) string {
	n = strings.Replace(n, "\n", "-", -1)
	n = strings.Replace(n, "\r", "-", -1)
	return n
}

func sanitizeValue(v string) string {
	v = strings.Replace(v, "\n", " ", -1)
	v = strings.Replace(v, "\r", " ", -1)
	v = strings.Replace(v, ";", " ", -1)
	return v
}

func unquoteCookieValue(v string) string {
	if len(v) > 1 && v[0] == '"' && v[len(v)-1] == '"' {
		return v[1 : len(v)-1]
	}
	return v
}

func isCookieByte(c byte) bool {
	switch {
	case c == 0x21, 0x23 <= c && c <= 0x2b, 0x2d <= c && c <= 0x3a,
		0x3c <= c && c <= 0x5b, 0x5d <= c && c <= 0x7e:
		return true
	}
	return false
}

func isCookieExpiresByte(c byte) (ok bool) {
	return isCookieByte(c) || c == ',' || c == ' '
}

func parseCookieValue(raw string) (string, bool) {
	return parseCookieValueUsing(raw, isCookieByte)
}

func parseCookieExpiresValue(raw string) (string, bool) {
	return parseCookieValueUsing(raw, isCookieExpiresByte)
}

func parseCookieValueUsing(raw string, validByte func(byte) bool) (string, bool) {
	raw = unquoteCookieValue(raw)
	for i := 0; i < len(raw); i++ {
		if !validByte(raw[i]) {
			return "", false
		}
	}
	return raw, true
}

func isCookieNameValid(raw string) bool {
	for _, c := range raw {
		if !isToken(byte(c)) {
			return false
		}
	}
	return true
}
