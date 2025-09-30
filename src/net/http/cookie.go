// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"errors"
	"fmt"
	"internal/godebug"
	"log"
	"net"
	"net/http/internal/ascii"
	"net/textproto"
	"strconv"
	"strings"
	"time"
)

var httpcookiemaxnum = godebug.New("httpcookiemaxnum")

// A Cookie represents an HTTP cookie as sent in the Set-Cookie header of an
// HTTP response or the Cookie header of an HTTP request.
//
// See https://tools.ietf.org/html/rfc6265 for details.
type Cookie struct {
	Name   string
	Value  string
	Quoted bool // indicates whether the Value was originally quoted

	Path       string    // optional
	Domain     string    // optional
	Expires    time.Time // optional
	RawExpires string    // for reading cookies only

	// MaxAge=0 means no 'Max-Age' attribute specified.
	// MaxAge<0 means delete cookie now, equivalently 'Max-Age: 0'
	// MaxAge>0 means Max-Age attribute present and given in seconds
	MaxAge      int
	Secure      bool
	HttpOnly    bool
	SameSite    SameSite
	Partitioned bool
	Raw         string
	Unparsed    []string // Raw text of unparsed attribute-value pairs
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

var (
	errBlankCookie            = errors.New("http: blank cookie")
	errEqualNotFoundInCookie  = errors.New("http: '=' not found in cookie")
	errInvalidCookieName      = errors.New("http: invalid cookie name")
	errInvalidCookieValue     = errors.New("http: invalid cookie value")
	errCookieNumLimitExceeded = errors.New("http: number of cookies exceeded limit")
)

const defaultCookieMaxNum = 3000

func cookieNumWithinMax(cookieNum int) bool {
	withinDefaultMax := cookieNum <= defaultCookieMaxNum
	if httpcookiemaxnum.Value() == "" {
		return withinDefaultMax
	}
	if customMax, err := strconv.Atoi(httpcookiemaxnum.Value()); err == nil {
		withinCustomMax := customMax == 0 || cookieNum <= customMax
		if withinDefaultMax != withinCustomMax {
			httpcookiemaxnum.IncNonDefault()
		}
		return withinCustomMax
	}
	return withinDefaultMax
}

// ParseCookie parses a Cookie header value and returns all the cookies
// which were set in it. Since the same cookie name can appear multiple times
// the returned Values can contain more than one value for a given key.
func ParseCookie(line string) ([]*Cookie, error) {
	if !cookieNumWithinMax(strings.Count(line, ";") + 1) {
		return nil, errCookieNumLimitExceeded
	}
	parts := strings.Split(textproto.TrimString(line), ";")
	if len(parts) == 1 && parts[0] == "" {
		return nil, errBlankCookie
	}
	cookies := make([]*Cookie, 0, len(parts))
	for _, s := range parts {
		s = textproto.TrimString(s)
		name, value, found := strings.Cut(s, "=")
		if !found {
			return nil, errEqualNotFoundInCookie
		}
		if !isToken(name) {
			return nil, errInvalidCookieName
		}
		value, quoted, found := parseCookieValue(value, true)
		if !found {
			return nil, errInvalidCookieValue
		}
		cookies = append(cookies, &Cookie{Name: name, Value: value, Quoted: quoted})
	}
	return cookies, nil
}

// ParseSetCookie parses a Set-Cookie header value and returns a cookie.
// It returns an error on syntax error.
func ParseSetCookie(line string) (*Cookie, error) {
	parts := strings.Split(textproto.TrimString(line), ";")
	if len(parts) == 1 && parts[0] == "" {
		return nil, errBlankCookie
	}
	parts[0] = textproto.TrimString(parts[0])
	name, value, ok := strings.Cut(parts[0], "=")
	if !ok {
		return nil, errEqualNotFoundInCookie
	}
	name = textproto.TrimString(name)
	if !isToken(name) {
		return nil, errInvalidCookieName
	}
	value, quoted, ok := parseCookieValue(value, true)
	if !ok {
		return nil, errInvalidCookieValue
	}
	c := &Cookie{
		Name:   name,
		Value:  value,
		Quoted: quoted,
		Raw:    line,
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
		val, _, ok = parseCookieValue(val, false)
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
		case "partitioned":
			c.Partitioned = true
			continue
		}
		c.Unparsed = append(c.Unparsed, parts[i])
	}
	return c, nil
}

// readSetCookies parses all "Set-Cookie" values from
// the header h and returns the successfully parsed Cookies.
//
// If the amount of cookies exceeds CookieNumLimit, and httpcookielimitnum
// GODEBUG option is not explicitly turned off, this function will silently
// fail and return an empty slice.
func readSetCookies(h Header) []*Cookie {
	cookieCount := len(h["Set-Cookie"])
	if cookieCount == 0 {
		return []*Cookie{}
	}
	// Cookie limit was unfortunately introduced at a later point in time.
	// As such, we can only fail by returning an empty slice rather than
	// explicit error.
	if !cookieNumWithinMax(cookieCount) {
		return []*Cookie{}
	}
	cookies := make([]*Cookie, 0, cookieCount)
	for _, line := range h["Set-Cookie"] {
		if cookie, err := ParseSetCookie(line); err == nil {
			cookies = append(cookies, cookie)
		}
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
	if c == nil || !isToken(c.Name) {
		return ""
	}
	// extraCookieLength derived from typical length of cookie attributes
	// see RFC 6265 Sec 4.1.
	const extraCookieLength = 110
	var b strings.Builder
	b.Grow(len(c.Name) + len(c.Value) + len(c.Domain) + len(c.Path) + extraCookieLength)
	b.WriteString(c.Name)
	b.WriteRune('=')
	b.WriteString(sanitizeCookieValue(c.Value, c.Quoted))

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
	if c.Partitioned {
		b.WriteString("; Partitioned")
	}
	return b.String()
}

// Valid reports whether the cookie is valid.
func (c *Cookie) Valid() error {
	if c == nil {
		return errors.New("http: nil Cookie")
	}
	if !isToken(c.Name) {
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
	if c.Partitioned {
		if !c.Secure {
			return errors.New("http: partitioned cookies must be set with Secure")
		}
	}
	return nil
}

// readCookies parses all "Cookie" values from the header h and
// returns the successfully parsed Cookies.
//
// If filter isn't empty, only cookies of that name are returned.
//
// If the amount of cookies exceeds CookieNumLimit, and httpcookielimitnum
// GODEBUG option is not explicitly turned off, this function will silently
// fail and return an empty slice.
func readCookies(h Header, filter string) []*Cookie {
	lines := h["Cookie"]
	if len(lines) == 0 {
		return []*Cookie{}
	}

	// Cookie limit was unfortunately introduced at a later point in time.
	// As such, we can only fail by returning an empty slice rather than
	// explicit error.
	cookieCount := 0
	for _, line := range lines {
		cookieCount += strings.Count(line, ";") + 1
	}
	if !cookieNumWithinMax(cookieCount) {
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
			if !isToken(name) {
				continue
			}
			if filter != "" && filter != name {
				continue
			}
			val, quoted, ok := parseCookieValue(val, true)
			if !ok {
				continue
			}
			cookies = append(cookies, &Cookie{Name: name, Value: val, Quoted: quoted})
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
// It receives a quoted bool indicating whether the value was originally
// quoted.
// https://tools.ietf.org/html/rfc6265#section-4.1.1
//
//	cookie-value      = *cookie-octet / ( DQUOTE *cookie-octet DQUOTE )
//	cookie-octet      = %x21 / %x23-2B / %x2D-3A / %x3C-5B / %x5D-7E
//	          ; US-ASCII characters excluding CTLs,
//	          ; whitespace DQUOTE, comma, semicolon,
//	          ; and backslash
//
// We loosen this as spaces and commas are common in cookie values
// thus we produce a quoted cookie-value if v contains commas or spaces.
// See https://golang.org/issue/7243 for the discussion.
func sanitizeCookieValue(v string, quoted bool) string {
	v = sanitizeOrWarn("Cookie.Value", validCookieValueByte, v)
	if len(v) == 0 {
		return v
	}
	if strings.ContainsAny(v, " ,") || quoted {
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

// parseCookieValue parses a cookie value according to RFC 6265.
// If allowDoubleQuote is true, parseCookieValue will consider that it
// is parsing the cookie-value;
// otherwise, it will consider that it is parsing a cookie-av value
// (cookie attribute-value).
//
// It returns the parsed cookie value, a boolean indicating whether the
// parsing was successful, and a boolean indicating whether the parsed
// value was enclosed in double quotes.
func parseCookieValue(raw string, allowDoubleQuote bool) (value string, quoted, ok bool) {
	// Strip the quotes, if present.
	if allowDoubleQuote && len(raw) > 1 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
		quoted = true
	}
	for i := 0; i < len(raw); i++ {
		if !validCookieValueByte(raw[i]) {
			return "", quoted, false
		}
	}
	return raw, quoted, true
}
