// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP Request reading and parsing.

// The http package implements parsing of HTTP requests, replies,
// and URLs and provides an extensible HTTP server and a basic
// HTTP client.
package http

import (
	"bufio"
	"bytes"
	"container/vector"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

const (
	maxLineLength  = 1024 // assumed < bufio.DefaultBufSize
	maxValueLength = 1024
	maxHeaderLines = 1024
	chunkSize      = 4 << 10 // 4 KB chunks
)

// HTTP request parsing errors.
type ProtocolError struct {
	os.ErrorString
}

var (
	ErrLineTooLong          = &ProtocolError{"header line too long"}
	ErrHeaderTooLong        = &ProtocolError{"header too long"}
	ErrShortBody            = &ProtocolError{"entity body too short"}
	ErrNotSupported         = &ProtocolError{"feature not supported"}
	ErrUnexpectedTrailer    = &ProtocolError{"trailer header without chunked transfer encoding"}
	ErrMissingContentLength = &ProtocolError{"missing ContentLength in HEAD response"}
)

type badStringError struct {
	what string
	str  string
}

func (e *badStringError) String() string { return fmt.Sprintf("%s %q", e.what, e.str) }

// A Request represents a parsed HTTP request header.
type Request struct {
	Method     string // GET, POST, PUT, etc.
	RawURL     string // The raw URL given in the request.
	URL        *URL   // Parsed URL.
	Proto      string // "HTTP/1.0"
	ProtoMajor int    // 1
	ProtoMinor int    // 0

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
	Header map[string]string

	// The message body.
	Body io.Reader

	// Whether to close the connection after replying to this request.
	Close bool

	// The host on which the URL is sought.
	// Per RFC 2616, this is either the value of the Host: header
	// or the host name given in the URL itself.
	Host string

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
	Referer string

	// The User-Agent: header string, if sent in the request.
	UserAgent string

	// The parsed form. Only available after ParseForm is called.
	Form map[string][]string
}

// ProtoAtLeast returns whether the HTTP protocol used
// in the request is at least major.minor.
func (r *Request) ProtoAtLeast(major, minor int) bool {
	return r.ProtoMajor > major ||
		r.ProtoMajor == major && r.ProtoMinor >= minor
}

// Return value if nonempty, def otherwise.
func valueOrDefault(value, def string) string {
	if value != "" {
		return value
	}
	return def
}

const defaultUserAgent = "Go http package"

// Write writes an HTTP/1.1 request -- header and body -- in wire format.
// This method consults the following fields of req:
//	URL
//	Method (defaults to "GET")
//	UserAgent (defaults to defaultUserAgent)
//	Referer
//	Header
//	Body
//
// If Body is present, "Transfer-Encoding: chunked" is forced as a header.
func (req *Request) Write(w io.Writer) os.Error {
	uri := urlEscape(req.URL.Path, false)
	if req.URL.RawQuery != "" {
		uri += "?" + req.URL.RawQuery
	}

	fmt.Fprintf(w, "%s %s HTTP/1.1\r\n", valueOrDefault(req.Method, "GET"), uri)
	fmt.Fprintf(w, "Host: %s\r\n", req.URL.Host)
	fmt.Fprintf(w, "User-Agent: %s\r\n", valueOrDefault(req.UserAgent, defaultUserAgent))

	if req.Referer != "" {
		fmt.Fprintf(w, "Referer: %s\r\n", req.Referer)
	}

	if req.Body != nil {
		// Force chunked encoding
		req.Header["Transfer-Encoding"] = "chunked"
	}

	// TODO: split long values?  (If so, should share code with Conn.Write)
	// TODO: if Header includes values for Host, User-Agent, or Referer, this
	// may conflict with the User-Agent or Referer headers we add manually.
	// One solution would be to remove the Host, UserAgent, and Referer fields
	// from Request, and introduce Request methods along the lines of
	// Response.{GetHeader,AddHeader} and string constants for "Host",
	// "User-Agent" and "Referer".
	for k, v := range req.Header {
		// Host, User-Agent, and Referer were sent from structure fields
		// above; ignore them if they also appear in req.Header.
		if k == "Host" || k == "User-Agent" || k == "Referer" {
			continue
		}
		io.WriteString(w, k+": "+v+"\r\n")
	}

	io.WriteString(w, "\r\n")

	if req.Body != nil {
		buf := make([]byte, chunkSize)
	Loop:
		for {
			var nr, nw int
			var er, ew os.Error
			if nr, er = req.Body.Read(buf); nr > 0 {
				if er == nil || er == os.EOF {
					fmt.Fprintf(w, "%x\r\n", nr)
					nw, ew = w.Write(buf[0:nr])
					fmt.Fprint(w, "\r\n")
				}
			}
			switch {
			case er != nil:
				if er == os.EOF {
					break Loop
				}
				return er
			case ew != nil:
				return ew
			case nw < nr:
				return io.ErrShortWrite
			}
		}
		// last-chunk CRLF
		fmt.Fprint(w, "0\r\n\r\n")
	}

	return nil
}

// Read a line of bytes (up to \n) from b.
// Give up if the line exceeds maxLineLength.
// The returned bytes are a pointer into storage in
// the bufio, so they are only valid until the next bufio read.
func readLineBytes(b *bufio.Reader) (p []byte, err os.Error) {
	if p, err = b.ReadSlice('\n'); err != nil {
		// We always know when EOF is coming.
		// If the caller asked for a line, there should be a line.
		if err == os.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	if len(p) >= maxLineLength {
		return nil, ErrLineTooLong
	}

	// Chop off trailing white space.
	var i int
	for i = len(p); i > 0; i-- {
		if c := p[i-1]; c != ' ' && c != '\r' && c != '\t' && c != '\n' {
			break
		}
	}
	return p[0:i], nil
}

// readLineBytes, but convert the bytes into a string.
func readLine(b *bufio.Reader) (s string, err os.Error) {
	p, e := readLineBytes(b)
	if e != nil {
		return "", e
	}
	return string(p), nil
}

var colon = []byte{':'}

// Read a key/value pair from b.
// A key/value has the form Key: Value\r\n
// and the Value can continue on multiple lines if each continuation line
// starts with a space.
func readKeyValue(b *bufio.Reader) (key, value string, err os.Error) {
	line, e := readLineBytes(b)
	if e != nil {
		return "", "", e
	}
	if len(line) == 0 {
		return "", "", nil
	}

	// Scan first line for colon.
	i := bytes.Index(line, colon)
	if i < 0 {
		goto Malformed
	}

	key = string(line[0:i])
	if strings.Index(key, " ") >= 0 {
		// Key field has space - no good.
		goto Malformed
	}

	// Skip initial space before value.
	for i++; i < len(line); i++ {
		if line[i] != ' ' {
			break
		}
	}
	value = string(line[i:])

	// Look for extension lines, which must begin with space.
	for {
		c, e := b.ReadByte()
		if c != ' ' {
			if e != os.EOF {
				b.UnreadByte()
			}
			break
		}

		// Eat leading space.
		for c == ' ' {
			if c, e = b.ReadByte(); e != nil {
				if e == os.EOF {
					e = io.ErrUnexpectedEOF
				}
				return "", "", e
			}
		}
		b.UnreadByte()

		// Read the rest of the line and add to value.
		if line, e = readLineBytes(b); e != nil {
			return "", "", e
		}
		value += " " + string(line)

		if len(value) >= maxValueLength {
			return "", "", &badStringError{"value too long for key", key}
		}
	}
	return key, value, nil

Malformed:
	return "", "", &badStringError{"malformed header line", string(line)}
}

// Convert decimal at s[i:len(s)] to integer,
// returning value, string position where the digits stopped,
// and whether there was a valid number (digits, not too big).
func atoi(s string, i int) (n, i1 int, ok bool) {
	const Big = 1000000
	if i >= len(s) || s[i] < '0' || s[i] > '9' {
		return 0, 0, false
	}
	n = 0
	for ; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
		n = n*10 + int(s[i]-'0')
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
	major, i, ok := atoi(vers, 5)
	if !ok || i >= len(vers) || vers[i] != '.' {
		return 0, 0, false
	}
	var minor int
	minor, i, ok = atoi(vers, i+1)
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
		return t
	}

	// canonicalize: first letter upper case
	// and upper case after each dash.
	// (Host, User-Agent, If-Modified-Since).
	// HTTP headers are ASCII only, so no Unicode issues.
	a := strings.Bytes(s)
	upper := true
	for i, v := range a {
		if upper && 'a' <= v && v <= 'z' {
			a[i] = v + 'A' - 'a'
		}
		if !upper && 'A' <= v && v <= 'Z' {
			a[i] = v + 'a' - 'A'
		}
		upper = false
		if v == '-' {
			upper = true
		}
	}
	t := string(a)
	cmap[s] = t
	return t
}

type chunkedReader struct {
	r   *bufio.Reader
	n   uint64 // unread bytes in chunk
	err os.Error
}

func newChunkedReader(r *bufio.Reader) *chunkedReader {
	return &chunkedReader{r: r}
}

func (cr *chunkedReader) beginChunk() {
	// chunk-size CRLF
	var line string
	line, cr.err = readLine(cr.r)
	if cr.err != nil {
		return
	}
	cr.n, cr.err = strconv.Btoui64(line, 16)
	if cr.err != nil {
		return
	}
	if cr.n == 0 {
		// trailer CRLF
		for {
			line, cr.err = readLine(cr.r)
			if cr.err != nil {
				return
			}
			if line == "" {
				break
			}
		}
		cr.err = os.EOF
	}
}

func (cr *chunkedReader) Read(b []uint8) (n int, err os.Error) {
	if cr.err != nil {
		return 0, cr.err
	}
	if cr.n == 0 {
		cr.beginChunk()
		if cr.err != nil {
			return 0, cr.err
		}
	}
	if uint64(len(b)) > cr.n {
		b = b[0:cr.n]
	}
	n, cr.err = cr.r.Read(b)
	cr.n -= uint64(n)
	if cr.n == 0 && cr.err == nil {
		// end of chunk (CRLF)
		b := make([]byte, 2)
		if _, cr.err = io.ReadFull(cr.r, b); cr.err == nil {
			if b[0] != '\r' || b[1] != '\n' {
				cr.err = os.NewError("malformed chunked encoding")
			}
		}
	}
	return n, cr.err
}

// ReadRequest reads and parses a request from b.
func ReadRequest(b *bufio.Reader) (req *Request, err os.Error) {
	req = new(Request)

	// First line: GET /index.html HTTP/1.0
	var s string
	if s, err = readLine(b); err != nil {
		return nil, err
	}

	var f []string
	if f = strings.Split(s, " ", 3); len(f) < 3 {
		return nil, &badStringError{"malformed HTTP request", s}
	}
	req.Method, req.RawURL, req.Proto = f[0], f[1], f[2]
	var ok bool
	if req.ProtoMajor, req.ProtoMinor, ok = parseHTTPVersion(req.Proto); !ok {
		return nil, &badStringError{"malformed HTTP version", req.Proto}
	}

	if req.URL, err = ParseURL(req.RawURL); err != nil {
		return nil, err
	}

	// Subsequent lines: Key: value.
	nheader := 0
	req.Header = make(map[string]string)
	for {
		var key, value string
		if key, value, err = readKeyValue(b); err != nil {
			return nil, err
		}
		if key == "" {
			break
		}
		if nheader++; nheader >= maxHeaderLines {
			return nil, ErrHeaderTooLong
		}

		key = CanonicalHeaderKey(key)

		// RFC 2616 says that if you send the same header key
		// multiple times, it has to be semantically equivalent
		// to concatenating the values separated by commas.
		oldvalue, present := req.Header[key]
		if present {
			req.Header[key] = oldvalue + "," + value
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
	if v, present := req.Header["Host"]; present {
		if req.URL.Host == "" {
			req.Host = v
		}
		req.Header["Host"] = "", false
	}

	// RFC2616: Should treat
	//	Pragma: no-cache
	// like
	//	Cache-Control: no-cache
	if v, present := req.Header["Pragma"]; present && v == "no-cache" {
		if _, presentcc := req.Header["Cache-Control"]; !presentcc {
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
		req.Header["Referer"] = "", false
	}
	if v, present := req.Header["User-Agent"]; present {
		req.UserAgent = v
		req.Header["User-Agent"] = "", false
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

	// A message body exists when either Content-Length or Transfer-Encoding
	// headers are present. Transfer-Encoding trumps Content-Length.
	if v, present := req.Header["Transfer-Encoding"]; present && v == "chunked" {
		req.Body = newChunkedReader(b)
	} else if v, present := req.Header["Content-Length"]; present {
		length, err := strconv.Btoui64(v, 10)
		if err != nil {
			return nil, &badStringError{"invalid Content-Length", v}
		}
		// TODO: limit the Content-Length. This is an easy DoS vector.
		raw := make([]byte, length)
		n, err := b.Read(raw)
		if err != nil || uint64(n) < length {
			return nil, ErrShortBody
		}
		req.Body = bytes.NewBuffer(raw)
	}

	return req, nil
}

func parseForm(m map[string][]string, query string) (err os.Error) {
	data := make(map[string]*vector.StringVector)
	for _, kv := range strings.Split(query, "&", 0) {
		kvPair := strings.Split(kv, "=", 2)

		var key, value string
		var e os.Error
		key, e = URLUnescape(kvPair[0])
		if e == nil && len(kvPair) > 1 {
			value, e = URLUnescape(kvPair[1])
		}
		if e != nil {
			err = e
		}

		vec, ok := data[key]
		if !ok {
			vec = new(vector.StringVector)
			data[key] = vec
		}
		vec.Push(value)
	}

	for k, vec := range data {
		m[k] = vec.Data()
	}

	return
}

// ParseForm parses the request body as a form for POST requests, or the raw query for GET requests.
// It is idempotent.
func (r *Request) ParseForm() (err os.Error) {
	if r.Form != nil {
		return
	}
	r.Form = make(map[string][]string)

	var query string
	switch r.Method {
	case "GET":
		query = r.URL.RawQuery
	case "POST":
		if r.Body == nil {
			return os.ErrorString("missing form body")
		}
		ct, _ := r.Header["Content-Type"]
		switch strings.Split(ct, ";", 2)[0] {
		case "text/plain", "application/x-www-form-urlencoded", "":
			var b []byte
			if b, err = ioutil.ReadAll(r.Body); err != nil {
				return err
			}
			query = string(b)
		// TODO(dsymonds): Handle multipart/form-data
		default:
			return &badStringError{"unknown Content-Type", ct}
		}
	}
	return parseForm(r.Form, query)
}

// FormValue returns the first value for the named component of the query.
// FormValue calls ParseForm if necessary.
func (r *Request) FormValue(key string) string {
	if r.Form == nil {
		r.ParseForm()
	}
	if vs, ok := r.Form[key]; ok && len(vs) > 0 {
		return vs[0]
	}
	return ""
}
