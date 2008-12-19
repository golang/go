// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP Request reading and parsing.

package http

import (
	"bufio";
	"http";
	"os";
	"strings"
)

const (
	MaxLineLength = 1024;	// assumed < bufio.DefaultBufSize
	MaxValueLength = 1024;
	MaxHeaderLines = 1024;
)

export var (
	LineTooLong = os.NewError("http header line too long");
	ValueTooLong = os.NewError("http header value too long");
	HeaderTooLong = os.NewError("http header too long");
	BadHeader = os.NewError("malformed http header");
	BadRequest = os.NewError("invalid http request");
	BadHTTPVersion = os.NewError("unsupported http version");
)

// HTTP Request
export type Request struct {
	method string;		// GET, PUT,etc.
	rawurl string;
	url *URL;		// URI after GET, PUT etc.
	proto string;	// "HTTP/1.0"
	pmajor int;	// 1
	pminor int;	// 0

	header map[string] string;

	close bool;
	host string;
	referer string;
	useragent string;
}

// Read a line of bytes (up to \n) from b.
// Give up if the line exceeds MaxLineLength.
// The returned bytes are a pointer into storage in
// the bufio, so they are only valid until the next bufio read.
func ReadLineBytes(b *bufio.BufRead) (p []byte, err *os.Error) {
	if p, err = b.ReadLineSlice('\n'); err != nil {
		return nil, err
	}
	if len(p) >= MaxLineLength {
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

// ReadLineByte, but convert the bytes into a string.
func ReadLine(b *bufio.BufRead) (s string, err *os.Error) {
	p, e := ReadLineBytes(b);
	if e != nil {
		return "", e
	}
	return string(p), nil
}

// Read a key/value pair from b.
// A key/value has the form Key: Value\r\n
// and the Value can continue on multiple lines if each continuation line
// starts with a space.
func ReadKeyValue(b *bufio.BufRead) (key, value string, err *os.Error) {
	line, e := ReadLineBytes(b);
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
				if line, e = ReadLineBytes(b); e != nil {
					return "", "", e
				}
				value += " " + string(line);

				if len(value) >= MaxValueLength {
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
func ParseHTTPVersion(vers string) (int, int, bool) {
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

// Read and parse a request from b.
export func ReadRequest(b *bufio.BufRead) (req *Request, err *os.Error) {
	req = new(*Request);

	// First line: GET /index.html HTTP/1.0
	var s string;
	if s, err = ReadLine(b); err != nil {
		return nil, err
	}

	var f []string;
	if f = strings.split(s, " "); len(f) != 3 {
		return nil, BadRequest
	}
	req.method, req.rawurl, req.proto = f[0], f[1], f[2];
	var ok bool;
	if req.pmajor, req.pminor, ok = ParseHTTPVersion(req.proto); !ok {
		return nil, BadHTTPVersion
	}

	if req.url, err = ParseURL(req.rawurl); err != nil {
		return nil, err
	}

	// Subsequent lines: Key: value.
	nheader := 0;
	req.header = new(map[string] string);
	for {
		var key, value string;
		if key, value, err = ReadKeyValue(b); err != nil {
			return nil, err
		}
		if key == "" {
			break
		}
		if nheader++; nheader >= MaxHeaderLines {
			return nil, HeaderTooLong
		}

		// RFC 2616 says that if you send the same header key
		// multiple times, it has to be semantically equivalent
		// to concatenating the values separated by commas.
		oldvalue, present := req.header[key];
		if present {
			req.header[key] = oldvalue+","+value
		} else {
			req.header[key] = value
		}
	}

	// RFC2616: Must treat
	//	GET /index.html HTTP/1.1
	//	Host: www.google.com
	// and
	//	GET http://www.google.com/index.html HTTP/1.1
	//	Host: doesntmatter
	// the same.  In the second case, any Host line is ignored.
	if v, have := req.header["Host"]; have && req.url.host == "" {
		req.host = v
	}

	// RFC2616: Should treat
	//	Pragma: no-cache
	// like
	//	Cache-control: no-cache
	if v, have := req.header["Pragma"]; have && v == "no-cache" {
		if cc, havecc := req.header["Cache-control"]; !havecc {
			req.header["Cache-control"] = "no-cache"
		}
	}

	// Determine whether to hang up after sending the reply.
	if req.pmajor < 1 || (req.pmajor == 1 && req.pminor < 1) {
		req.close = true
	} else if v, have := req.header["Connection"]; have {
		// TODO: Should split on commas, toss surrounding white space,
		// and check each field.
		if v == "close" {
			req.close = true
		}
	}

	// Pull out useful fields as a convenience to clients.
	if v, have := req.header["Referer"]; have {
		req.referer = v
	}
	if v, have := req.header["User-Agent"]; have {
		req.useragent = v
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
