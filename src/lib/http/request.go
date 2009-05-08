// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP Request reading and parsing.

// The http package implements parsing of HTTP requests and URLs
// and provides an extensible HTTP server.
//
// In the future it should also implement parsing of HTTP replies
// and provide methods to fetch URLs via HTTP.
package http

import (
	"bufio";
	"http";
	"io";
	"os";
	"strings"
)

const (
	maxLineLength = 1024;	// assumed < bufio.DefaultBufSize
	maxValueLength = 1024;
	maxHeaderLines = 1024;
)

// HTTP request parsing errors.
var (
	LineTooLong = os.NewError("http header line too long");
	ValueTooLong = os.NewError("http header value too long");
	HeaderTooLong = os.NewError("http header too long");
	BadHeader = os.NewError("malformed http header");
	BadRequest = os.NewError("invalid http request");
	BadHTTPVersion = os.NewError("unsupported http version");
)

// A Request represents a parsed HTTP request header.
type Request struct {
	Method string;		// GET, PUT,etc.
	RawUrl string;		// The raw URL given in the request.
	Url *URL;		// URL after GET, PUT etc.
	Proto string;	// "HTTP/1.0"
	ProtoMajor int;	// 1
	ProtoMinor int;	// 0

	// A header mapping request lines to their values.
	// If the header says
	//
	//	Accept-Language: en-us
	//	accept-encoding: gzip, deflate
	//	Connection: keep-alive
	//
	// then
	//
	//	Header = map[string]string{
	//		"Accept-Encoding": "en-us",
	//		"Accept-Language": "gzip, deflate",
	//		"Connection": "keep-alive"
	//	}
	//
	// HTTP defines that header names are case-insensitive.
	// The request parser implements this by canonicalizing the
	// name, making the first character and any characters
	// following a hyphen uppercase and the rest lowercase.
	Header map[string] string;

	// Whether to close the connection after replying to this request.
	Close bool;

	// The host on which the URL is sought.
	// Per RFC 2616, this is either the value of the Host: header
	// or the host name given in the URL itself.
	Host string;

	// The referring URL, if sent in the request.
	//
	// Referer is misspelled as in the request itself,
	// a mistake from the earliest days of HTTP.
	// This value can also be fetched from the Header map
	// as Header["Referer"]; the benefit of making it
	// available as a structure field is that the compiler
	// can diagnose programs that use the alternate
	// (correct English) spelling req.Referrer but cannot
	// diagnose programs that use Header["Referrer"].
	Referer string;

	// The User-Agent: header string, if sent in the request.
	UserAgent string;
}

// ProtoAtLeast returns whether the HTTP protocol used
// in the request is at least major.minor.
func (r *Request) ProtoAtLeast(major, minor int) bool {
	return r.ProtoMajor > major ||
		r.ProtoMajor == major && r.ProtoMinor >= minor
}

// Read a line of bytes (up to \n) from b.
// Give up if the line exceeds maxLineLength.
// The returned bytes are a pointer into storage in
// the bufio, so they are only valid until the next bufio read.
func readLineBytes(b *bufio.Reader) (p []byte, err os.Error) {
	if p, err = b.ReadLineSlice('\n'); err != nil {
		return nil, err
	}
	if len(p) >= maxLineLength {
		return nil, LineTooLong
	}

	// Chop off trailing white space.
	var i int;
	for i = len(p); i > 0; i-- {
		if c := p[i-1]; c != ' ' && c != '\r' && c != '\t' && c != '\n' {
			break
		}
	}
	return p[0:i], nil
}

// readLineBytes, but convert the bytes into a string.
func readLine(b *bufio.Reader) (s string, err os.Error) {
	p, e := readLineBytes(b);
	if e != nil {
		return "", e
	}
	return string(p), nil
}

// Read a key/value pair from b.
// A key/value has the form Key: Value\r\n
// and the Value can continue on multiple lines if each continuation line
// starts with a space.
func readKeyValue(b *bufio.Reader) (key, value string, err os.Error) {
	line, e := readLineBytes(b);
	if e != nil {
		return "", "", e
	}
	if len(line) == 0 {
		return "", "", nil
	}

	// Scan first line for colon.
	for i := 0; i < len(line); i++ {
		switch line[i] {
		case ' ':
			// Key field has space - no good.
			return "", "", BadHeader;
		case ':':
			key = string(line[0:i]);
			// Skip initial space before value.
			for i++; i < len(line); i++ {
				if line[i] != ' ' {
					break
				}
			}
			value = string(line[i:len(line)]);

			// Look for extension lines, which must begin with space.
			for {
				var c byte;

				if c, e = b.ReadByte(); e != nil {
					return "", "", e
				}
				if c != ' ' {
					// Not leading space; stop.
					b.UnreadByte();
					break
				}

				// Eat leading space.
				for c == ' ' {
					if c, e = b.ReadByte(); e != nil {
						return "", "", e
					}
				}
				b.UnreadByte();

				// Read the rest of the line and add to value.
				if line, e = readLineBytes(b); e != nil {
					return "", "", e
				}
				value += " " + string(line);

				if len(value) >= maxValueLength {
					return "", "", ValueTooLong
				}
			}
			return key, value, nil
		}
	}

	// Line ended before space or colon.
	return "", "", BadHeader;
}

// Convert decimal at s[i:len(s)] to integer,
// returning value, string position where the digits stopped,
// and whether there was a valid number (digits, not too big).
func atoi(s string, i int) (n, i1 int, ok bool) {
	const Big = 1000000;
	if i >= len(s) || s[i] < '0' || s[i] > '9' {
		return 0, 0, false
	}
	n = 0;
	for ; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
		n = n*10 + int(s[i]-'0');
		if n > Big {
			return 0, 0, false
		}
	}
	return n, i, true
}

// Parse HTTP version: "HTTP/1.2" -> (1, 2, true).
func parseHTTPVersion(vers string) (int, int, bool) {
	if vers[0:5] != "HTTP/" {
		return 0, 0, false
	}
	major, i, ok := atoi(vers, 5);
	if !ok || i >= len(vers) || vers[i] != '.' {
		return 0, 0, false
	}
	var minor int;
	minor, i, ok = atoi(vers, i+1);
	if !ok || i != len(vers) {
		return 0, 0, false
	}
	return major, minor, true
}

var cmap = make(map[string]string)

// CanonicalHeaderKey returns the canonical format of the
// HTTP header key s.  The canonicalization converts the first
// letter and any letter following a hyphen to upper case;
// the rest are converted to lowercase.  For example, the
// canonical key for "accept-encoding" is "Accept-Encoding".
func CanonicalHeaderKey(s string) string {
	if t, ok := cmap[s]; ok {
		return t;
	}

	// canonicalize: first letter upper case
	// and upper case after each dash.
	// (Host, User-Agent, If-Modified-Since).
	// HTTP headers are ASCII only, so no Unicode issues.
	a := io.StringBytes(s);
	upper := true;
	for i,v := range a {
		if upper && 'a' <= v && v <= 'z' {
			a[i] = v + 'A' - 'a';
		}
		if !upper && 'A' <= v && v <= 'Z' {
			a[i] = v + 'a' - 'A';
		}
		upper = false;
		if v == '-' {
			upper = true;
		}
	}
	t := string(a);
	cmap[s] = t;
	return t;
}

// ReadRequest reads and parses a request from b.
func ReadRequest(b *bufio.Reader) (req *Request, err os.Error) {
	req = new(Request);

	// First line: GET /index.html HTTP/1.0
	var s string;
	if s, err = readLine(b); err != nil {
		return nil, err
	}

	var f []string;
	if f = strings.Split(s, " "); len(f) != 3 {
		return nil, BadRequest
	}
	req.Method, req.RawUrl, req.Proto = f[0], f[1], f[2];
	var ok bool;
	if req.ProtoMajor, req.ProtoMinor, ok = parseHTTPVersion(req.Proto); !ok {
		return nil, BadHTTPVersion
	}

	if req.Url, err = ParseURL(req.RawUrl); err != nil {
		return nil, err
	}

	// Subsequent lines: Key: value.
	nheader := 0;
	req.Header = make(map[string] string);
	for {
		var key, value string;
		if key, value, err = readKeyValue(b); err != nil {
			return nil, err
		}
		if key == "" {
			break
		}
		if nheader++; nheader >= maxHeaderLines {
			return nil, HeaderTooLong
		}

		key = CanonicalHeaderKey(key);

		// RFC 2616 says that if you send the same header key
		// multiple times, it has to be semantically equivalent
		// to concatenating the values separated by commas.
		oldvalue, present := req.Header[key];
		if present {
			req.Header[key] = oldvalue+","+value
		} else {
			req.Header[key] = value
		}
	}

	// RFC2616: Must treat
	//	GET /index.html HTTP/1.1
	//	Host: www.google.com
	// and
	//	GET http://www.google.com/index.html HTTP/1.1
	//	Host: doesntmatter
	// the same.  In the second case, any Host line is ignored.
	if v, present := req.Header["Host"]; present && req.Url.Host == "" {
		req.Host = v
	}

	// RFC2616: Should treat
	//	Pragma: no-cache
	// like
	//	Cache-Control: no-cache
	if v, present := req.Header["Pragma"]; present && v == "no-cache" {
		if cc, presentcc := req.Header["Cache-Control"]; !presentcc {
			req.Header["Cache-Control"] = "no-cache"
		}
	}

	// Determine whether to hang up after sending the reply.
	if req.ProtoMajor < 1 || (req.ProtoMajor == 1 && req.ProtoMinor < 1) {
		req.Close = true
	} else if v, present := req.Header["Connection"]; present {
		// TODO: Should split on commas, toss surrounding white space,
		// and check each field.
		if v == "close" {
			req.Close = true
		}
	}

	// Pull out useful fields as a convenience to clients.
	if v, present := req.Header["Referer"]; present {
		req.Referer = v
	}
	if v, present := req.Header["User-Agent"]; present {
		req.UserAgent = v
	}

	// TODO: Parse specific header values:
	//	Accept
	//	Accept-Encoding
	//	Accept-Language
	//	Authorization
	//	Cache-Control
	//	Connection
	//	Date
	//	Expect
	//	From
	//	If-Match
	//	If-Modified-Since
	//	If-None-Match
	//	If-Range
	//	If-Unmodified-Since
	//	Max-Forwards
	//	Proxy-Authorization
	//	Referer [sic]
	//	TE (transfer-codings)
	//	Trailer
	//	Transfer-Encoding
	//	Upgrade
	//	User-Agent
	//	Via
	//	Warning

	return req, nil;
}
