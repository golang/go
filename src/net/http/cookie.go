// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"errors"
	"fmt"
	"log"
	"net"
	"net/http/internal/ascii"
	"net/textproto"
	"strconv"
	"strings"
	"time"
)

// A Cookie represents an HTTP cookie as sent in the Set-Cookie header of an
// HTTP response or the Cookie header of an HTTP request.
//
// See https://tools.ietf.org/html/rfc6265 for details.
type Cookie struct {
	Name  string
	Value string

	Path       string    // optional
	Domain     string    // optional
	Expires    time.Time // optional
	RawExpires string    // for reading cookies only

	// MaxAge=0 means no 'Max-Age' attribute specified.
	// MaxAge<0 means delete cookie now, equivalently 'Max-Age: 0'
	// MaxAge>0 means Max-Age attribute present and given in seconds
	MaxAge   int
	Secure   bool
	HttpOnly bool
	SameSite SameSite
	Raw      string
	Unparsed []string // Raw text of unparsed attribute-value pairs
}

// SameSite allows a server to define a cookie attribute making it impossible for
// the browser to send this cookie along with cross-site requests. The main
// goal is to mitigate the risk of cross-origin information leakage, and provide
// some protection against cross-site request forgery attacks.
//
// See https://tools.ietf.org/html/draft-ietf-httpbis-cookie-same-site-00 for details.
type SameSite int

const (
	SameSiteDefaultMode SameSite = iota + 1
	SameSiteLaxMode
	SameSiteStrictMode
	SameSiteNoneMode
)

// readSetCookies parses all "Set-Cookie" values from
// the header h and returns the successfully parsed Cookies.
func readSetCookies(h Header) []*Cookie {
	cookieCount := len(h["Set-Cookie"])
	if cookieCount == 0 {
		return []*Cookie{}
	}
	cookies := make([]*Cookie, 0, cookieCount)
	for _, line := range h["Set-Cookie"] {
		parts := strings.Split(textproto.TrimString(line), ";")
		if len(parts) == 1 && parts[0] == "" {
			continue
		}
		parts[0] = textproto.TrimString(parts[0])
		name, value, ok := strings.Cut(parts[0], "=")
		if !ok {
			continue
		}
		name = textproto.TrimString(name)
		if !isCookieNameValid(name) {
			continue
		}
		value, ok = parseCookieValue(value, true)
		if !ok {
			continue
		}
		c := &Cookie{
			Name:  name,
			Value: value,
			Raw:   line,
		}
		for i := 1; i < len(parts); i++ {
			parts[i] = textproto.TrimString(parts[i])
			if len(parts[i]) == 0 {
				continue
			}

			attr, val, _ := strings.Cut(parts[i], "=")
			lowerAttr, isASCII := ascii.ToLower(attr)
			if !isASCII {
				continue
			}
			val, ok = parseCookieValue(val, false)
			if !ok {
				c.Unparsed = append(c.Unparsed, parts[i])
				continue
			}

			switch lowerAttr {
			case "samesite":
				lowerVal, ascii := ascii.ToLower(val)
				if !ascii {
					c.SameSite = SameSiteDefaultMode
					continue
				}
				switch lowerVal {
				case "lax":
					c.SameSite = SameSiteLaxMode
				case "strict":
					c.SameSite = SameSiteStrictMode
				case "none":
					c.SameSite = SameSiteNoneMode
				default:
					c.SameSite = SameSiteDefaultMode
				}
				continue
			case "secure":
				c.Secure = true
				continue
			case "httponly":
				c.HttpOnly = true
				continue
			case "domain":
				c.Domain = val
				continue
			case "max-age":
				secs, err := strconv.Atoi(val)
				if err != nil || secs != 0 && val[0] == '0' {
					break
				}
				if secs <= 0 {
					secs = -1
				}
				c.MaxAge = secs
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
				c.Expires = exptime.UTC()
				continue
			case "path":
				c.Path = val
				continue
			}
			c.Unparsed = append(c.Unparsed, parts[i])
		}
		cookies = append(cookies, c)
	}
	return cookies
}

// SetCookie adds a Set-Cookie header to the provided [ResponseWriter]'s headers.
// The provided cookie must have a valid Name. Invalid cookies may be
// silently dropped.
func SetCookie(w ResponseWriter, cookie *Cookie) {
	if v := cookie.String(); v != "" {
		w.Header().Add("Set-Cookie", v)
	}
}

// String returns the serialization of the cookie for use in a [Cookie]
// header (if only Name and Value are set) or a Set-Cookie response
// header (if other fields are set).
// If c is nil or c.Name is invalid, the empty string is returned.
func (c *Cookie) String() string {
	if c == nil || !isCookieNameValid(c.Name) {
		return ""
	}
	// extraCookieLength derived from typical length of cookie attributes
	// see RFC 6265 Sec 4.1.
	const extraCookieLength = 110
	var b strings.Builder
	b.Grow(len(c.Name) + len(c.Value) + len(c.Domain) + len(c.Path) + extraCookieLength)
	b.WriteString(c.Name)
	b.WriteRune('=')
	b.WriteString(sanitizeCookieValue(c.Value))

	if len(c.Path) > 0 {
		b.WriteString("; Path=")
		b.WriteString(sanitizeCookiePath(c.Path))
	}
	if len(c.Domain) > 0 {
		if validCookieDomain(c.Domain) {
			// A c.Domain containing illegal characters is not
			// sanitized but simply dropped which turns the cookie
			// into a host-only cookie. A leading dot is okay
			// but won't be sent.
			d := c.Domain
			if d[0] == '.' {
				d = d[1:]
			}
			b.WriteString("; Domain=")
			b.WriteString(d)
		} else {
			log.Printf("net/http: invalid Cookie.Domain %q; dropping domain attribute", c.Domain)
		}
	}
	var buf [len(TimeFormat)]byte
	if validCookieExpires(c.Expires) {
		b.WriteString("; Expires=")
		b.Write(c.Expires.UTC().AppendFormat(buf[:0], TimeFormat))
	}
	if c.MaxAge > 0 {
		b.WriteString("; Max-Age=")
		b.Write(strconv.AppendInt(buf[:0], int64(c.MaxAge), 10))
	} else if c.MaxAge < 0 {
		b.WriteString("; Max-Age=0")
	}
	if c.HttpOnly {
		b.WriteString("; HttpOnly")
	}
	if c.Secure {
		b.WriteString("; Secure")
	}
	switch c.SameSite {
	case SameSiteDefaultMode:
		// Skip, default mode is obtained by not emitting the attribute.
	case SameSiteNoneMode:
		b.WriteString("; SameSite=None")
	case SameSiteLaxMode:
		b.WriteString("; SameSite=Lax")
	case SameSiteStrictMode:
		b.WriteString("; SameSite=Strict")
	}
	return b.String()
}

// Valid reports whether the cookie is valid.
func (c *Cookie) Valid() error {
	if c == nil {
		return errors.New("http: nil Cookie")
	}
	if !isCookieNameValid(c.Name) {
		return errors.New("http: invalid Cookie.Name")
	}
	if !c.Expires.IsZero() && !validCookieExpires(c.Expires) {
		return errors.New("http: invalid Cookie.Expires")
	}
	for i := 0; i < len(c.Value); i++ {
		if !validCookieValueByte(c.Value[i]) {
			return fmt.Errorf("http: invalid byte %q in Cookie.Value", c.Value[i])
		}
	}
	if len(c.Path) > 0 {
		for i := 0; i < len(c.Path); i++ {
			if !validCookiePathByte(c.Path[i]) {
				return fmt.Errorf("http: invalid byte %q in Cookie.Path", c.Path[i])
			}
		}
	}
	if len(c.Domain) > 0 {
		if !validCookieDomain(c.Domain) {
			return errors.New("http: invalid Cookie.Domain")
		}
	}
	return nil
}

// readCookies parses all "Cookie" values from the header h and
// returns the successfully parsed Cookies.
//
// if filter isn't empty, only cookies of that name are returned.
func readCookies(h Header, filter string) []*Cookie {
	lines := h["Cookie"]
	if len(lines) == 0 {
		return []*Cookie{}
	}

	cookies := make([]*Cookie, 0, len(lines)+strings.Count(lines[0], ";"))
	for _, line := range lines {
		line = textproto.TrimString(line)

		var part string
		for len(line) > 0 { // continue since we have rest
			part, line, _ = strings.Cut(line, ";")
			part = textproto.TrimString(part)
			if part == "" {
				continue
			}
			name, val, _ := strings.Cut(part, "=")
			name = textproto.TrimString(name)
			if !isCookieNameValid(name) {
				continue
			}
			if filter != "" && filter != name {
				continue
			}
			val, ok := parseCookieValue(val, true)
			if !ok {
				continue
			}
			cookies = append(cookies, &Cookie{Name: name, Value: val})
		}
	}
	return cookies
}

// validCookieDomain reports whether v is a valid cookie domain-value.
func validCookieDomain(v string) bool {
	if isCookieDomainName(v) {
		return true
	}
	if net.ParseIP(v) != nil && !strings.Contains(v, ":") {
		return true
	}
	return false
}

// validCookieExpires reports whether v is a valid cookie expires-value.
func validCookieExpires(t time.Time) bool {
	// IETF RFC 6265 Section 5.1.1.5, the year must not be less than 1601
	return t.Year() >= 1601
}

// isCookieDomainName reports whether s is a valid domain name or a valid
// domain name with a leading dot '.'.  It is almost a direct copy of
// package net's isDomainName.
func isCookieDomainName(s string) bool {
	if len(s) == 0 {
		return false
	}
	if len(s) > 255 {
		return false
	}

	if s[0] == '.' {
		// A cookie a domain attribute may start with a leading dot.
		s = s[1:]
	}
	last := byte('.')
	ok := false // Ok once we've seen a letter.
	partlen := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		default:
			return false
		case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
			// No '_' allowed here (in contrast to package net).
			ok = true
			partlen++
		case '0' <= c && c <= '9':
			// fine
			partlen++
		case c == '-':
			// Byte before dash cannot be dot.
			if last == '.' {
				return false
			}
			partlen++
		case c == '.':
			// Byte before dot cannot be dot, dash.
			if last == '.' || last == '-' {
				return false
			}
			if partlen > 63 || partlen == 0 {
				return false
			}
			partlen = 0
		}
		last = c
	}
	if last == '-' || partlen > 63 {
		return false
	}

	return ok
}

var cookieNameSanitizer = strings.NewReplacer("\n", "-", "\r", "-")

func sanitizeCookieName(n string) string {
	return cookieNameSanitizer.Replace(n)
}

// sanitizeCookieValue produces a suitable cookie-value from v.
// https://tools.ietf.org/html/rfc6265#section-4.1.1
//
//	cookie-value      = *cookie-octet / ( DQUOTE *cookie-octet DQUOTE )
//	cookie-octet      = %x21 / %x23-2B / %x2D-3A / %x3C-5B / %x5D-7E
//	          ; US-ASCII characters excluding CTLs,
//	          ; whitespace DQUOTE, comma, semicolon,
//	          ; and backslash
//
// We loosen this as spaces and commas are common in cookie values
// but we produce a quoted cookie-value if and only if v contains
// commas or spaces.
// See https://golang.org/issue/7243 for the discussion.
func sanitizeCookieValue(v string) string {
	v = sanitizeOrWarn("Cookie.Value", validCookieValueByte, v)
	if len(v) == 0 {
		return v
	}
	if strings.ContainsAny(v, " ,") {
		return `"` + v + `"`
	}
	return v
}

func validCookieValueByte(b byte) bool {
	return 0x20 <= b && b < 0x7f && b != '"' && b != ';' && b != '\\'
}

// path-av           = "Path=" path-value
// path-value        = <any CHAR except CTLs or ";">
func sanitizeCookiePath(v string) string {
	return sanitizeOrWarn("Cookie.Path", validCookiePathByte, v)
}

func validCookiePathByte(b byte) bool {
	return 0x20 <= b && b < 0x7f && b != ';'
}

func sanitizeOrWarn(fieldName string, valid func(byte) bool, v string) string {
	ok := true
	for i := 0; i < len(v); i++ {
		if valid(v[i]) {
			continue
		}
		log.Printf("net/http: invalid byte %q in %s; dropping invalid bytes", v[i], fieldName)
		ok = false
		break
	}
	if ok {
		return v
	}
	buf := make([]byte, 0, len(v))
	for i := 0; i < len(v); i++ {
		if b := v[i]; valid(b) {
			buf = append(buf, b)
		}
	}
	return string(buf)
}

func parseCookieValue(raw string, allowDoubleQuote bool) (string, bool) {
	// Strip the quotes, if present.
	if allowDoubleQuote && len(raw) > 1 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
	}
	for i := 0; i < len(raw); i++ {
		if !validCookieValueByte(raw[i]) {
			return "", false
		}
	}
	return raw, true
}

func isCookieNameValid(raw string) bool {
	if raw == "" {
		return false
	}
	return strings.IndexFunc(raw, isNotToken) < 0
}
