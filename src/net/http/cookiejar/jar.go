// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cookiejar implements an in-memory RFC 6265-compliant http.CookieJar.
package cookiejar

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"
)

// PublicSuffixList provides the public suffix of a domain. For example:
//      - the public suffix of "example.com" is "com",
//      - the public suffix of "foo1.foo2.foo3.co.uk" is "co.uk", and
//      - the public suffix of "bar.pvt.k12.ma.us" is "pvt.k12.ma.us".
//
// Implementations of PublicSuffixList must be safe for concurrent use by
// multiple goroutines.
//
// An implementation that always returns "" is valid and may be useful for
// testing but it is not secure: it means that the HTTP server for foo.com can
// set a cookie for bar.com.
//
// A public suffix list implementation is in the package
// golang.org/x/net/publicsuffix.
type PublicSuffixList interface {
	// PublicSuffix returns the public suffix of domain.
	//
	// TODO: specify which of the caller and callee is responsible for IP
	// addresses, for leading and trailing dots, for case sensitivity, and
	// for IDN/Punycode.
	PublicSuffix(domain string) string

	// String returns a description of the source of this public suffix
	// list. The description will typically contain something like a time
	// stamp or version number.
	String() string
}

// Options are the options for creating a new Jar.
type Options struct {
	// PublicSuffixList is the public suffix list that determines whether
	// an HTTP server can set a cookie for a domain.
	//
	// A nil value is valid and may be useful for testing but it is not
	// secure: it means that the HTTP server for foo.co.uk can set a cookie
	// for bar.co.uk.
	PublicSuffixList PublicSuffixList
}

// Jar implements the http.CookieJar interface from the net/http package.
type Jar struct {
	psList PublicSuffixList

	// mu locks the remaining fields.
	mu sync.Mutex

	// entries is a set of entries, keyed by their eTLD+1 and subkeyed by
	// their name/domain/path.
	entries map[string]map[string]entry

	// nextSeqNum is the next sequence number assigned to a new cookie
	// created SetCookies.
	nextSeqNum uint64
}

// New returns a new cookie jar. A nil *Options is equivalent to a zero
// Options.
func New(o *Options) (*Jar, error) {
	jar := &Jar{
		entries: make(map[string]map[string]entry),
	}
	if o != nil {
		jar.psList = o.PublicSuffixList
	}
	return jar, nil
}

// entry is the internal representation of a cookie.
//
// This struct type is not used outside of this package per se, but the exported
// fields are those of RFC 6265.
type entry struct {
	Name       string
	Value      string
	Domain     string
	Path       string
	SameSite   string
	Secure     bool
	HttpOnly   bool
	Persistent bool
	HostOnly   bool
	Expires    time.Time
	Creation   time.Time
	LastAccess time.Time

	// seqNum is a sequence number so that Cookies returns cookies in a
	// deterministic order, even for cookies that have equal Path length and
	// equal Creation time. This simplifies testing.
	seqNum uint64
}

// id returns the domain;path;name triple of e as an id.
func (e *entry) id() string {
	return fmt.Sprintf("%s;%s;%s", e.Domain, e.Path, e.Name)
}

// shouldSend determines whether e's cookie qualifies to be included in a
// request to host/path. It is the caller's responsibility to check if the
// cookie is expired.
func (e *entry) shouldSend(https bool, host, path string) bool {
	return e.domainMatch(host) && e.pathMatch(path) && (https || !e.Secure)
}

// domainMatch implements "domain-match" of RFC 6265 section 5.1.3.
func (e *entry) domainMatch(host string) bool {
	if e.Domain == host {
		return true
	}
	return !e.HostOnly && hasDotSuffix(host, e.Domain)
}

// pathMatch implements "path-match" according to RFC 6265 section 5.1.4.
func (e *entry) pathMatch(requestPath string) bool {
	if requestPath == e.Path {
		return true
	}
	if strings.HasPrefix(requestPath, e.Path) {
		if e.Path[len(e.Path)-1] == '/' {
			return true // The "/any/" matches "/any/path" case.
		} else if requestPath[len(e.Path)] == '/' {
			return true // The "/any" matches "/any/path" case.
		}
	}
	return false
}

// hasDotSuffix reports whether s ends in "."+suffix.
func hasDotSuffix(s, suffix string) bool {
	return len(s) > len(suffix) && s[len(s)-len(suffix)-1] == '.' && s[len(s)-len(suffix):] == suffix
}

// Cookies implements the Cookies method of the http.CookieJar interface.
//
// It returns an empty slice if the URL's scheme is not HTTP or HTTPS.
func (j *Jar) Cookies(u *url.URL) (cookies []*http.Cookie) {
	return j.cookies(u, time.Now())
}

// cookies is like Cookies but takes the current time as a parameter.
func (j *Jar) cookies(u *url.URL, now time.Time) (cookies []*http.Cookie) {
	if u.Scheme != "http" && u.Scheme != "https" {
		return cookies
	}
	host, err := canonicalHost(u.Host)
	if err != nil {
		return cookies
	}
	key := jarKey(host, j.psList)

	j.mu.Lock()
	defer j.mu.Unlock()

	submap := j.entries[key]
	if submap == nil {
		return cookies
	}

	https := u.Scheme == "https"
	path := u.Path
	if path == "" {
		path = "/"
	}

	modified := false
	var selected []entry
	for id, e := range submap {
		if e.Persistent && !e.Expires.After(now) {
			delete(submap, id)
			modified = true
			continue
		}
		if !e.shouldSend(https, host, path) {
			continue
		}
		e.LastAccess = now
		submap[id] = e
		selected = append(selected, e)
		modified = true
	}
	if modified {
		if len(submap) == 0 {
			delete(j.entries, key)
		} else {
			j.entries[key] = submap
		}
	}

	// sort according to RFC 6265 section 5.4 point 2: by longest
	// path and then by earliest creation time.
	sort.Slice(selected, func(i, j int) bool {
		s := selected
		if len(s[i].Path) != len(s[j].Path) {
			return len(s[i].Path) > len(s[j].Path)
		}
		if !s[i].Creation.Equal(s[j].Creation) {
			return s[i].Creation.Before(s[j].Creation)
		}
		return s[i].seqNum < s[j].seqNum
	})
	for _, e := range selected {
		cookies = append(cookies, &http.Cookie{Name: e.Name, Value: e.Value})
	}

	return cookies
}

// SetCookies implements the SetCookies method of the http.CookieJar interface.
//
// It does nothing if the URL's scheme is not HTTP or HTTPS.
func (j *Jar) SetCookies(u *url.URL, cookies []*http.Cookie) {
	j.setCookies(u, cookies, time.Now())
}

// setCookies is like SetCookies but takes the current time as parameter.
func (j *Jar) setCookies(u *url.URL, cookies []*http.Cookie, now time.Time) {
	if len(cookies) == 0 {
		return
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return
	}
	host, err := canonicalHost(u.Host)
	if err != nil {
		return
	}
	key := jarKey(host, j.psList)
	defPath := defaultPath(u.Path)

	j.mu.Lock()
	defer j.mu.Unlock()

	submap := j.entries[key]

	modified := false
	for _, cookie := range cookies {
		e, remove, err := j.newEntry(cookie, now, defPath, host)
		if err != nil {
			continue
		}
		id := e.id()
		if remove {
			if submap != nil {
				if _, ok := submap[id]; ok {
					delete(submap, id)
					modified = true
				}
			}
			continue
		}
		if submap == nil {
			submap = make(map[string]entry)
		}

		if old, ok := submap[id]; ok {
			e.Creation = old.Creation
			e.seqNum = old.seqNum
		} else {
			e.Creation = now
			e.seqNum = j.nextSeqNum
			j.nextSeqNum++
		}
		e.LastAccess = now
		submap[id] = e
		modified = true
	}

	if modified {
		if len(submap) == 0 {
			delete(j.entries, key)
		} else {
			j.entries[key] = submap
		}
	}
}

// canonicalHost strips port from host if present and returns the canonicalized
// host name.
func canonicalHost(host string) (string, error) {
	var err error
	host = strings.ToLower(host)
	if hasPort(host) {
		host, _, err = net.SplitHostPort(host)
		if err != nil {
			return "", err
		}
	}
	if strings.HasSuffix(host, ".") {
		// Strip trailing dot from fully qualified domain names.
		host = host[:len(host)-1]
	}
	return toASCII(host)
}

// hasPort reports whether host contains a port number. host may be a host
// name, an IPv4 or an IPv6 address.
func hasPort(host string) bool {
	colons := strings.Count(host, ":")
	if colons == 0 {
		return false
	}
	if colons == 1 {
		return true
	}
	return host[0] == '[' && strings.Contains(host, "]:")
}

// jarKey returns the key to use for a jar.
func jarKey(host string, psl PublicSuffixList) string {
	if isIP(host) {
		return host
	}

	var i int
	if psl == nil {
		i = strings.LastIndex(host, ".")
		if i <= 0 {
			return host
		}
	} else {
		suffix := psl.PublicSuffix(host)
		if suffix == host {
			return host
		}
		i = len(host) - len(suffix)
		if i <= 0 || host[i-1] != '.' {
			// The provided public suffix list psl is broken.
			// Storing cookies under host is a safe stopgap.
			return host
		}
		// Only len(suffix) is used to determine the jar key from
		// here on, so it is okay if psl.PublicSuffix("www.buggy.psl")
		// returns "com" as the jar key is generated from host.
	}
	prevDot := strings.LastIndex(host[:i-1], ".")
	return host[prevDot+1:]
}

// isIP reports whether host is an IP address.
func isIP(host string) bool {
	return net.ParseIP(host) != nil
}

// defaultPath returns the directory part of an URL's path according to
// RFC 6265 section 5.1.4.
func defaultPath(path string) string {
	if len(path) == 0 || path[0] != '/' {
		return "/" // Path is empty or malformed.
	}

	i := strings.LastIndex(path, "/") // Path starts with "/", so i != -1.
	if i == 0 {
		return "/" // Path has the form "/abc".
	}
	return path[:i] // Path is either of form "/abc/xyz" or "/abc/xyz/".
}

// newEntry creates an entry from a http.Cookie c. now is the current time and
// is compared to c.Expires to determine deletion of c. defPath and host are the
// default-path and the canonical host name of the URL c was received from.
//
// remove records whether the jar should delete this cookie, as it has already
// expired with respect to now. In this case, e may be incomplete, but it will
// be valid to call e.id (which depends on e's Name, Domain and Path).
//
// A malformed c.Domain will result in an error.
func (j *Jar) newEntry(c *http.Cookie, now time.Time, defPath, host string) (e entry, remove bool, err error) {
	e.Name = c.Name

	if c.Path == "" || c.Path[0] != '/' {
		e.Path = defPath
	} else {
		e.Path = c.Path
	}

	e.Domain, e.HostOnly, err = j.domainAndType(host, c.Domain)
	if err != nil {
		return e, false, err
	}

	// MaxAge takes precedence over Expires.
	if c.MaxAge < 0 {
		return e, true, nil
	} else if c.MaxAge > 0 {
		e.Expires = now.Add(time.Duration(c.MaxAge) * time.Second)
		e.Persistent = true
	} else {
		if c.Expires.IsZero() {
			e.Expires = endOfTime
			e.Persistent = false
		} else {
			if !c.Expires.After(now) {
				return e, true, nil
			}
			e.Expires = c.Expires
			e.Persistent = true
		}
	}

	e.Value = c.Value
	e.Secure = c.Secure
	e.HttpOnly = c.HttpOnly

	switch c.SameSite {
	case http.SameSiteDefaultMode:
		e.SameSite = "SameSite"
	case http.SameSiteStrictMode:
		e.SameSite = "SameSite=Strict"
	case http.SameSiteLaxMode:
		e.SameSite = "SameSite=Lax"
	}

	return e, false, nil
}

var (
	errIllegalDomain   = errors.New("cookiejar: illegal cookie domain attribute")
	errMalformedDomain = errors.New("cookiejar: malformed cookie domain attribute")
	errNoHostname      = errors.New("cookiejar: no host name available (IP only)")
)

// endOfTime is the time when session (non-persistent) cookies expire.
// This instant is representable in most date/time formats (not just
// Go's time.Time) and should be far enough in the future.
var endOfTime = time.Date(9999, 12, 31, 23, 59, 59, 0, time.UTC)

// domainAndType determines the cookie's domain and hostOnly attribute.
func (j *Jar) domainAndType(host, domain string) (string, bool, error) {
	if domain == "" {
		// No domain attribute in the SetCookie header indicates a
		// host cookie.
		return host, true, nil
	}

	if isIP(host) {
		// According to RFC 6265 domain-matching includes not being
		// an IP address.
		// TODO: This might be relaxed as in common browsers.
		return "", false, errNoHostname
	}

	// From here on: If the cookie is valid, it is a domain cookie (with
	// the one exception of a public suffix below).
	// See RFC 6265 section 5.2.3.
	if domain[0] == '.' {
		domain = domain[1:]
	}

	if len(domain) == 0 || domain[0] == '.' {
		// Received either "Domain=." or "Domain=..some.thing",
		// both are illegal.
		return "", false, errMalformedDomain
	}
	domain = strings.ToLower(domain)

	if domain[len(domain)-1] == '.' {
		// We received stuff like "Domain=www.example.com.".
		// Browsers do handle such stuff (actually differently) but
		// RFC 6265 seems to be clear here (e.g. section 4.1.2.3) in
		// requiring a reject.  4.1.2.3 is not normative, but
		// "Domain Matching" (5.1.3) and "Canonicalized Host Names"
		// (5.1.2) are.
		return "", false, errMalformedDomain
	}

	// See RFC 6265 section 5.3 #5.
	if j.psList != nil {
		if ps := j.psList.PublicSuffix(domain); ps != "" && !hasDotSuffix(domain, ps) {
			if host == domain {
				// This is the one exception in which a cookie
				// with a domain attribute is a host cookie.
				return host, true, nil
			}
			return "", false, errIllegalDomain
		}
	}

	// The domain must domain-match host: www.mycompany.com cannot
	// set cookies for .ourcompetitors.com.
	if host != domain && !hasDotSuffix(host, domain) {
		return "", false, errIllegalDomain
	}

	return domain, false, nil
}
