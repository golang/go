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
	"mime"
	"mime/multipart"
	"os"
	"strconv"
	"strings"
)

const (
	maxLineLength  = 4096 // assumed <= bufio.defaultBufSize
	maxValueLength = 4096
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
	ErrNotMultipart         = &ProtocolError{"request Content-Type isn't multipart/form-data"}
	ErrMissingBoundary      = &ProtocolError{"no multipart boundary param Content-Type"}
)

type badStringError struct {
	what string
	str  string
}

func (e *badStringError) String() string { return fmt.Sprintf("%s %q", e.what, e.str) }

var reqExcludeHeader = map[string]bool{
	"Host":              true,
	"User-Agent":        true,
	"Referer":           true,
	"Content-Length":    true,
	"Transfer-Encoding": true,
	"Trailer":           true,
}

// A Request represents a parsed HTTP request header.
type Request struct {
	Method     string // GET, POST, PUT, etc.
	RawURL     string // The raw URL given in the request.
	URL        *URL   // Parsed URL.
	Proto      string // "HTTP/1.0"
	ProtoMajor int    // 1
	ProtoMinor int    // 0

	// A header maps request lines to their values.
	// If the header says
	//
	//	accept-encoding: gzip, deflate
	//	Accept-Language: en-us
	//	Connection: keep-alive
	//
	// then
	//
	//	Header = map[string]string{
	//		"Accept-Encoding": "gzip, deflate",
	//		"Accept-Language": "en-us",
	//		"Connection": "keep-alive",
	//	}
	//
	// HTTP defines that header names are case-insensitive.
	// The request parser implements this by canonicalizing the
	// name, making the first character and any characters
	// following a hyphen uppercase and the rest lowercase.
	Header map[string]string

	// The message body.
	Body io.ReadCloser

	// ContentLength records the length of the associated content.
	// The value -1 indicates that the length is unknown.
	// Values >= 0 indicate that the given number of bytes may be read from Body.
	ContentLength int64

	// TransferEncoding lists the transfer encodings from outermost to innermost.
	// An empty list denotes the "identity" encoding.
	TransferEncoding []string

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

	// Trailer maps trailer keys to values.  Like for Header, if the
	// response has multiple trailer lines with the same key, they will be
	// concatenated, delimited by commas.
	Trailer map[string]string
}

// ProtoAtLeast returns whether the HTTP protocol used
// in the request is at least major.minor.
func (r *Request) ProtoAtLeast(major, minor int) bool {
	return r.ProtoMajor > major ||
		r.ProtoMajor == major && r.ProtoMinor >= minor
}

// MultipartReader returns a MIME multipart reader if this is a
// multipart/form-data POST request, else returns nil and an error.
func (r *Request) MultipartReader() (multipart.Reader, os.Error) {
	v, ok := r.Header["Content-Type"]
	if !ok {
		return nil, ErrNotMultipart
	}
	d, params := mime.ParseMediaType(v)
	if d != "multipart/form-data" {
		return nil, ErrNotMultipart
	}
	boundary, ok := params["boundary"]
	if !ok {
		return nil, ErrMissingBoundary
	}
	return multipart.NewReader(r.Body, boundary), nil
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
//	Host
//	RawURL, if non-empty, or else URL
//	Method (defaults to "GET")
//	UserAgent (defaults to defaultUserAgent)
//	Referer
//	Header
//	Body
//
// If Body is present, Write forces "Transfer-Encoding: chunked" as a header
// and then closes Body when finished sending it.
func (req *Request) Write(w io.Writer) os.Error {
	host := req.Host
	if host == "" {
		host = req.URL.Host
	}

	uri := req.RawURL
	if uri == "" {
		uri = valueOrDefault(urlEscape(req.URL.Path, encodePath), "/")
		if req.URL.RawQuery != "" {
			uri += "?" + req.URL.RawQuery
		}
	}

	fmt.Fprintf(w, "%s %s HTTP/1.1\r\n", valueOrDefault(req.Method, "GET"), uri)

	// Header lines
	fmt.Fprintf(w, "Host: %s\r\n", host)
	fmt.Fprintf(w, "User-Agent: %s\r\n", valueOrDefault(req.UserAgent, defaultUserAgent))
	if req.Referer != "" {
		fmt.Fprintf(w, "Referer: %s\r\n", req.Referer)
	}

	// Process Body,ContentLength,Close,Trailer
	tw, err := newTransferWriter(req)
	if err != nil {
		return err
	}
	err = tw.WriteHeader(w)
	if err != nil {
		return err
	}

	// TODO: split long values?  (If so, should share code with Conn.Write)
	// TODO: if Header includes values for Host, User-Agent, or Referer, this
	// may conflict with the User-Agent or Referer headers we add manually.
	// One solution would be to remove the Host, UserAgent, and Referer fields
	// from Request, and introduce Request methods along the lines of
	// Response.{GetHeader,AddHeader} and string constants for "Host",
	// "User-Agent" and "Referer".
	err = writeSortedKeyValue(w, req.Header, reqExcludeHeader)
	if err != nil {
		return err
	}

	io.WriteString(w, "\r\n")

	// Write body and trailer
	err = tw.WriteBody(w)
	if err != nil {
		return err
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
		} else if err == bufio.ErrBufferFull {
			err = ErrLineTooLong
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
	if strings.Contains(key, " ") {
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
	if len(vers) < 5 || vers[0:5] != "HTTP/" {
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

// CanonicalHeaderKey returns the canonical format of the
// HTTP header key s.  The canonicalization converts the first
// letter and any letter following a hyphen to upper case;
// the rest are converted to lowercase.  For example, the
// canonical key for "accept-encoding" is "Accept-Encoding".
func CanonicalHeaderKey(s string) string {
	// canonicalize: first letter upper case
	// and upper case after each dash.
	// (Host, User-Agent, If-Modified-Since).
	// HTTP headers are ASCII only, so no Unicode issues.
	var a []byte
	upper := true
	for i := 0; i < len(s); i++ {
		v := s[i]
		if upper && 'a' <= v && v <= 'z' {
			if a == nil {
				a = []byte(s)
			}
			a[i] = v + 'A' - 'a'
		}
		if !upper && 'A' <= v && v <= 'Z' {
			if a == nil {
				a = []byte(s)
			}
			a[i] = v + 'a' - 'A'
		}
		upper = false
		if v == '-' {
			upper = true
		}
	}
	if a != nil {
		return string(a)
	}
	return s
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

	if req.URL, err = ParseRequestURL(req.RawURL); err != nil {
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
	req.Host = req.URL.Host
	if req.Host == "" {
		req.Host = req.Header["Host"]
	}
	req.Header["Host"] = "", false

	fixPragmaCacheControl(req.Header)

	// Pull out useful fields as a convenience to clients.
	req.Referer = req.Header["Referer"]
	req.Header["Referer"] = "", false

	req.UserAgent = req.Header["User-Agent"]
	req.Header["User-Agent"] = "", false

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

	err = readTransfer(req, b)
	if err != nil {
		return nil, err
	}

	return req, nil
}

// ParseQuery parses the URL-encoded query string and returns
// a map listing the values specified for each key.
// ParseQuery always returns a non-nil map containing all the
// valid query parameters found; err describes the first decoding error
// encountered, if any.
func ParseQuery(query string) (m map[string][]string, err os.Error) {
	m = make(map[string][]string)
	err = parseQuery(m, query)
	return
}

func parseQuery(m map[string][]string, query string) (err os.Error) {
	for _, kv := range strings.Split(query, "&", -1) {
		if len(kv) == 0 {
			continue
		}
		kvPair := strings.Split(kv, "=", 2)

		var key, value string
		var e os.Error
		key, e = URLUnescape(kvPair[0])
		if e == nil && len(kvPair) > 1 {
			value, e = URLUnescape(kvPair[1])
		}
		if e != nil {
			err = e
			continue
		}
		vec := vector.StringVector(m[key])
		vec.Push(value)
		m[key] = vec
	}
	return err
}

// ParseForm parses the request body as a form for POST requests, or the raw query for GET requests.
// It is idempotent.
func (r *Request) ParseForm() (err os.Error) {
	if r.Form != nil {
		return
	}

	r.Form = make(map[string][]string)
	if r.URL != nil {
		err = parseQuery(r.Form, r.URL.RawQuery)
	}
	if r.Method == "POST" {
		if r.Body == nil {
			return os.ErrorString("missing form body")
		}
		ct := r.Header["Content-Type"]
		switch strings.Split(ct, ";", 2)[0] {
		case "text/plain", "application/x-www-form-urlencoded", "":
			b, e := ioutil.ReadAll(r.Body)
			if e != nil {
				if err == nil {
					err = e
				}
				break
			}
			e = parseQuery(r.Form, string(b))
			if err == nil {
				err = e
			}
		// TODO(dsymonds): Handle multipart/form-data
		default:
			return &badStringError{"unknown Content-Type", ct}
		}
	}
	return err
}

// FormValue returns the first value for the named component of the query.
// FormValue calls ParseForm if necessary.
func (r *Request) FormValue(key string) string {
	if r.Form == nil {
		r.ParseForm()
	}
	if vs := r.Form[key]; len(vs) > 0 {
		return vs[0]
	}
	return ""
}

func (r *Request) expectsContinue() bool {
	expectation, ok := r.Header["Expect"]
	return ok && strings.ToLower(expectation) == "100-continue"
}

func (r *Request) wantsHttp10KeepAlive() bool {
	if r.ProtoMajor != 1 || r.ProtoMinor != 0 {
		return false
	}
	value, exists := r.Header["Connection"]
	if !exists {
		return false
	}
	return strings.Contains(strings.ToLower(value), "keep-alive")
}
