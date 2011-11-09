// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP Request reading and parsing.

package http

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"mime"
	"mime/multipart"
	"net/textproto"
	"net/url"
	"strings"
)

const (
	maxValueLength   = 4096
	maxHeaderLines   = 1024
	chunkSize        = 4 << 10  // 4 KB chunks
	defaultMaxMemory = 32 << 20 // 32 MB
)

// ErrMissingFile is returned by FormFile when the provided file field name
// is either not present in the request or not a file field.
var ErrMissingFile = errors.New("http: no such file")

// HTTP request parsing errors.
type ProtocolError struct {
	ErrorString string
}

func (err *ProtocolError) Error() string { return err.ErrorString }

var (
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

func (e *badStringError) Error() string { return fmt.Sprintf("%s %q", e.what, e.str) }

// Headers that Request.Write handles itself and should be skipped.
var reqWriteExcludeHeader = map[string]bool{
	"Host":              true, // not in Header map anyway
	"User-Agent":        true,
	"Content-Length":    true,
	"Transfer-Encoding": true,
	"Trailer":           true,
}

// A Request represents an HTTP request received by a server
// or to be sent by a client.
type Request struct {
	Method string // GET, POST, PUT, etc.
	URL    *url.URL

	// The protocol version for incoming requests.
	// Outgoing requests always use HTTP/1.1.
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
	//	Header = map[string][]string{
	//		"Accept-Encoding": {"gzip, deflate"},
	//		"Accept-Language": {"en-us"},
	//		"Connection": {"keep-alive"},
	//	}
	//
	// HTTP defines that header names are case-insensitive.
	// The request parser implements this by canonicalizing the
	// name, making the first character and any characters
	// following a hyphen uppercase and the rest lowercase.
	Header Header

	// The message body.
	Body io.ReadCloser

	// ContentLength records the length of the associated content.
	// The value -1 indicates that the length is unknown.
	// Values >= 0 indicate that the given number of bytes may
	// be read from Body.
	// For outgoing requests, a value of 0 means unknown if Body is not nil.
	ContentLength int64

	// TransferEncoding lists the transfer encodings from outermost to
	// innermost. An empty list denotes the "identity" encoding.
	// TransferEncoding can usually be ignored; chunked encoding is
	// automatically added and removed as necessary when sending and
	// receiving requests.
	TransferEncoding []string

	// Close indicates whether to close the connection after
	// replying to this request.
	Close bool

	// The host on which the URL is sought.
	// Per RFC 2616, this is either the value of the Host: header
	// or the host name given in the URL itself.
	Host string

	// Form contains the parsed form data, including both the URL
	// field's query parameters and the POST or PUT form data.
	// This field is only available after ParseForm is called.
	// The HTTP client ignores Form and uses Body instead.
	Form url.Values

	// MultipartForm is the parsed multipart form, including file uploads.
	// This field is only available after ParseMultipartForm is called.
	// The HTTP client ignores MultipartForm and uses Body instead.
	MultipartForm *multipart.Form

	// Trailer maps trailer keys to values.  Like for Header, if the
	// response has multiple trailer lines with the same key, they will be
	// concatenated, delimited by commas.
	// For server requests, Trailer is only populated after Body has been
	// closed or fully consumed.
	// Trailer support is only partially complete.
	Trailer Header

	// RemoteAddr allows HTTP servers and other software to record
	// the network address that sent the request, usually for
	// logging. This field is not filled in by ReadRequest and
	// has no defined format. The HTTP server in this package
	// sets RemoteAddr to an "IP:port" address before invoking a
	// handler.
	// This field is ignored by the HTTP client.
	RemoteAddr string

	// TLS allows HTTP servers and other software to record
	// information about the TLS connection on which the request
	// was received. This field is not filled in by ReadRequest.
	// The HTTP server in this package sets the field for
	// TLS-enabled connections before invoking a handler;
	// otherwise it leaves the field nil.
	// This field is ignored by the HTTP client.
	TLS *tls.ConnectionState
}

// ProtoAtLeast returns whether the HTTP protocol used
// in the request is at least major.minor.
func (r *Request) ProtoAtLeast(major, minor int) bool {
	return r.ProtoMajor > major ||
		r.ProtoMajor == major && r.ProtoMinor >= minor
}

// UserAgent returns the client's User-Agent, if sent in the request.
func (r *Request) UserAgent() string {
	return r.Header.Get("User-Agent")
}

// Cookies parses and returns the HTTP cookies sent with the request.
func (r *Request) Cookies() []*Cookie {
	return readCookies(r.Header, "")
}

var ErrNoCookie = errors.New("http: named cookied not present")

// Cookie returns the named cookie provided in the request or
// ErrNoCookie if not found.
func (r *Request) Cookie(name string) (*Cookie, error) {
	for _, c := range readCookies(r.Header, name) {
		return c, nil
	}
	return nil, ErrNoCookie
}

// AddCookie adds a cookie to the request.  Per RFC 6265 section 5.4,
// AddCookie does not attach more than one Cookie header field.  That
// means all cookies, if any, are written into the same line,
// separated by semicolon.
func (r *Request) AddCookie(c *Cookie) {
	s := fmt.Sprintf("%s=%s", sanitizeName(c.Name), sanitizeValue(c.Value))
	if c := r.Header.Get("Cookie"); c != "" {
		r.Header.Set("Cookie", c+"; "+s)
	} else {
		r.Header.Set("Cookie", s)
	}
}

// Referer returns the referring URL, if sent in the request.
//
// Referer is misspelled as in the request itself, a mistake from the
// earliest days of HTTP.  This value can also be fetched from the
// Header map as Header["Referer"]; the benefit of making it available
// as a method is that the compiler can diagnose programs that use the
// alternate (correct English) spelling req.Referrer() but cannot
// diagnose programs that use Header["Referrer"].
func (r *Request) Referer() string {
	return r.Header.Get("Referer")
}

// multipartByReader is a sentinel value.
// Its presence in Request.MultipartForm indicates that parsing of the request
// body has been handed off to a MultipartReader instead of ParseMultipartFrom.
var multipartByReader = &multipart.Form{
	Value: make(map[string][]string),
	File:  make(map[string][]*multipart.FileHeader),
}

// MultipartReader returns a MIME multipart reader if this is a
// multipart/form-data POST request, else returns nil and an error.
// Use this function instead of ParseMultipartForm to
// process the request body as a stream.
func (r *Request) MultipartReader() (*multipart.Reader, error) {
	if r.MultipartForm == multipartByReader {
		return nil, errors.New("http: MultipartReader called twice")
	}
	if r.MultipartForm != nil {
		return nil, errors.New("http: multipart handled by ParseMultipartForm")
	}
	r.MultipartForm = multipartByReader
	return r.multipartReader()
}

func (r *Request) multipartReader() (*multipart.Reader, error) {
	v := r.Header.Get("Content-Type")
	if v == "" {
		return nil, ErrNotMultipart
	}
	d, params, err := mime.ParseMediaType(v)
	if err != nil || d != "multipart/form-data" {
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
//	URL
//	Method (defaults to "GET")
//	Header
//	ContentLength
//	TransferEncoding
//	Body
//
// If Body is present, Content-Length is <= 0 and TransferEncoding
// hasn't been set to "identity", Write adds "Transfer-Encoding:
// chunked" to the header. Body is closed after it is sent.
func (req *Request) Write(w io.Writer) error {
	return req.write(w, false, nil)
}

// WriteProxy is like Write but writes the request in the form
// expected by an HTTP proxy.  In particular, WriteProxy writes the
// initial Request-URI line of the request with an absolute URI, per
// section 5.1.2 of RFC 2616, including the scheme and host. In
// either case, WriteProxy also writes a Host header, using either
// req.Host or req.URL.Host.
func (req *Request) WriteProxy(w io.Writer) error {
	return req.write(w, true, nil)
}

// extraHeaders may be nil
func (req *Request) write(w io.Writer, usingProxy bool, extraHeaders Header) error {
	host := req.Host
	if host == "" {
		if req.URL == nil {
			return errors.New("http: Request.Write on Request with no Host or URL set")
		}
		host = req.URL.Host
	}

	urlStr := req.URL.RawPath
	if strings.HasPrefix(urlStr, "?") {
		urlStr = "/" + urlStr // Issue 2344
	}
	if urlStr == "" {
		urlStr = valueOrDefault(req.URL.RawPath, valueOrDefault(req.URL.EncodedPath(), "/"))
		if req.URL.RawQuery != "" {
			urlStr += "?" + req.URL.RawQuery
		}
		if usingProxy {
			if urlStr == "" || urlStr[0] != '/' {
				urlStr = "/" + urlStr
			}
			urlStr = req.URL.Scheme + "://" + host + urlStr
		}
	}
	// TODO(bradfitz): escape at least newlines in urlStr?

	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s %s HTTP/1.1\r\n", valueOrDefault(req.Method, "GET"), urlStr)

	// Header lines
	fmt.Fprintf(bw, "Host: %s\r\n", host)

	// Use the defaultUserAgent unless the Header contains one, which
	// may be blank to not send the header.
	userAgent := defaultUserAgent
	if req.Header != nil {
		if ua := req.Header["User-Agent"]; len(ua) > 0 {
			userAgent = ua[0]
		}
	}
	if userAgent != "" {
		fmt.Fprintf(bw, "User-Agent: %s\r\n", userAgent)
	}

	// Process Body,ContentLength,Close,Trailer
	tw, err := newTransferWriter(req)
	if err != nil {
		return err
	}
	err = tw.WriteHeader(bw)
	if err != nil {
		return err
	}

	// TODO: split long values?  (If so, should share code with Conn.Write)
	err = req.Header.WriteSubset(bw, reqWriteExcludeHeader)
	if err != nil {
		return err
	}

	if extraHeaders != nil {
		err = extraHeaders.Write(bw)
		if err != nil {
			return err
		}
	}

	io.WriteString(bw, "\r\n")

	// Write body and trailer
	err = tw.WriteBody(bw)
	if err != nil {
		return err
	}
	bw.Flush()
	return nil
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

// ParseHTTPVersion parses a HTTP version string.
// "HTTP/1.0" returns (1, 0, true).
func ParseHTTPVersion(vers string) (major, minor int, ok bool) {
	if len(vers) < 5 || vers[0:5] != "HTTP/" {
		return 0, 0, false
	}
	major, i, ok := atoi(vers, 5)
	if !ok || i >= len(vers) || vers[i] != '.' {
		return 0, 0, false
	}
	minor, i, ok = atoi(vers, i+1)
	if !ok || i != len(vers) {
		return 0, 0, false
	}
	return major, minor, true
}

// NewRequest returns a new Request given a method, URL, and optional body.
func NewRequest(method, urlStr string, body io.Reader) (*Request, error) {
	u, err := url.Parse(urlStr)
	if err != nil {
		return nil, err
	}
	rc, ok := body.(io.ReadCloser)
	if !ok && body != nil {
		rc = ioutil.NopCloser(body)
	}
	req := &Request{
		Method:     method,
		URL:        u,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(Header),
		Body:       rc,
		Host:       u.Host,
	}
	if body != nil {
		switch v := body.(type) {
		case *strings.Reader:
			req.ContentLength = int64(v.Len())
		case *bytes.Buffer:
			req.ContentLength = int64(v.Len())
		}
	}

	return req, nil
}

// SetBasicAuth sets the request's Authorization header to use HTTP
// Basic Authentication with the provided username and password.
//
// With HTTP Basic Authentication the provided username and password
// are not encrypted.
func (r *Request) SetBasicAuth(username, password string) {
	s := username + ":" + password
	r.Header.Set("Authorization", "Basic "+base64.StdEncoding.EncodeToString([]byte(s)))
}

// ReadRequest reads and parses a request from b.
func ReadRequest(b *bufio.Reader) (req *Request, err error) {

	tp := textproto.NewReader(b)
	req = new(Request)

	// First line: GET /index.html HTTP/1.0
	var s string
	if s, err = tp.ReadLine(); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}

	var f []string
	if f = strings.SplitN(s, " ", 3); len(f) < 3 {
		return nil, &badStringError{"malformed HTTP request", s}
	}
	var rawurl string
	req.Method, rawurl, req.Proto = f[0], f[1], f[2]
	var ok bool
	if req.ProtoMajor, req.ProtoMinor, ok = ParseHTTPVersion(req.Proto); !ok {
		return nil, &badStringError{"malformed HTTP version", req.Proto}
	}

	if req.URL, err = url.ParseRequest(rawurl); err != nil {
		return nil, err
	}

	// Subsequent lines: Key: value.
	mimeHeader, err := tp.ReadMIMEHeader()
	if err != nil {
		return nil, err
	}
	req.Header = Header(mimeHeader)

	// RFC2616: Must treat
	//	GET /index.html HTTP/1.1
	//	Host: www.google.com
	// and
	//	GET http://www.google.com/index.html HTTP/1.1
	//	Host: doesntmatter
	// the same.  In the second case, any Host line is ignored.
	req.Host = req.URL.Host
	if req.Host == "" {
		req.Host = req.Header.Get("Host")
	}
	req.Header.Del("Host")

	fixPragmaCacheControl(req.Header)

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

// MaxBytesReader is similar to io.LimitReader but is intended for
// limiting the size of incoming request bodies. In contrast to
// io.LimitReader, MaxBytesReader's result is a ReadCloser, returns a
// non-EOF error for a Read beyond the limit, and Closes the
// underlying reader when its Close method is called.
//
// MaxBytesReader prevents clients from accidentally or maliciously
// sending a large request and wasting server resources.
func MaxBytesReader(w ResponseWriter, r io.ReadCloser, n int64) io.ReadCloser {
	return &maxBytesReader{w: w, r: r, n: n}
}

type maxBytesReader struct {
	w       ResponseWriter
	r       io.ReadCloser // underlying reader
	n       int64         // max bytes remaining
	stopped bool
}

func (l *maxBytesReader) Read(p []byte) (n int, err error) {
	if l.n <= 0 {
		if !l.stopped {
			l.stopped = true
			if res, ok := l.w.(*response); ok {
				res.requestTooLarge()
			}
		}
		return 0, errors.New("http: request body too large")
	}
	if int64(len(p)) > l.n {
		p = p[:l.n]
	}
	n, err = l.r.Read(p)
	l.n -= int64(n)
	return
}

func (l *maxBytesReader) Close() error {
	return l.r.Close()
}

// ParseForm parses the raw query from the URL.
//
// For POST or PUT requests, it also parses the request body as a form.
// If the request Body's size has not already been limited by MaxBytesReader,
// the size is capped at 10MB.
//
// ParseMultipartForm calls ParseForm automatically.
// It is idempotent.
func (r *Request) ParseForm() (err error) {
	if r.Form != nil {
		return
	}
	if r.URL != nil {
		r.Form, err = url.ParseQuery(r.URL.RawQuery)
	}
	if r.Method == "POST" || r.Method == "PUT" {
		if r.Body == nil {
			return errors.New("missing form body")
		}
		ct := r.Header.Get("Content-Type")
		ct, _, err := mime.ParseMediaType(ct)
		switch {
		case ct == "application/x-www-form-urlencoded":
			var reader io.Reader = r.Body
			maxFormSize := int64(1<<63 - 1)
			if _, ok := r.Body.(*maxBytesReader); !ok {
				maxFormSize = int64(10 << 20) // 10 MB is a lot of text.
				reader = io.LimitReader(r.Body, maxFormSize+1)
			}
			b, e := ioutil.ReadAll(reader)
			if e != nil {
				if err == nil {
					err = e
				}
				break
			}
			if int64(len(b)) > maxFormSize {
				return errors.New("http: POST too large")
			}
			var newValues url.Values
			newValues, e = url.ParseQuery(string(b))
			if err == nil {
				err = e
			}
			if r.Form == nil {
				r.Form = make(url.Values)
			}
			// Copy values into r.Form. TODO: make this smoother.
			for k, vs := range newValues {
				for _, value := range vs {
					r.Form.Add(k, value)
				}
			}
		case ct == "multipart/form-data":
			// handled by ParseMultipartForm (which is calling us, or should be)
			// TODO(bradfitz): there are too many possible
			// orders to call too many functions here.
			// Clean this up and write more tests.
			// request_test.go contains the start of this,
			// in TestRequestMultipartCallOrder.
		default:
			return &badStringError{"unknown Content-Type", ct}
		}
	}
	return err
}

// ParseMultipartForm parses a request body as multipart/form-data.
// The whole request body is parsed and up to a total of maxMemory bytes of
// its file parts are stored in memory, with the remainder stored on
// disk in temporary files.
// ParseMultipartForm calls ParseForm if necessary.
// After one call to ParseMultipartForm, subsequent calls have no effect.
func (r *Request) ParseMultipartForm(maxMemory int64) error {
	if r.MultipartForm == multipartByReader {
		return errors.New("http: multipart handled by MultipartReader")
	}
	if r.Form == nil {
		err := r.ParseForm()
		if err != nil {
			return err
		}
	}
	if r.MultipartForm != nil {
		return nil
	}

	mr, err := r.multipartReader()
	if err == ErrNotMultipart {
		return nil
	} else if err != nil {
		return err
	}

	f, err := mr.ReadForm(maxMemory)
	if err != nil {
		return err
	}
	for k, v := range f.Value {
		r.Form[k] = append(r.Form[k], v...)
	}
	r.MultipartForm = f

	return nil
}

// FormValue returns the first value for the named component of the query.
// FormValue calls ParseMultipartForm and ParseForm if necessary.
func (r *Request) FormValue(key string) string {
	if r.Form == nil {
		r.ParseMultipartForm(defaultMaxMemory)
	}
	if vs := r.Form[key]; len(vs) > 0 {
		return vs[0]
	}
	return ""
}

// FormFile returns the first file for the provided form key.
// FormFile calls ParseMultipartForm and ParseForm if necessary.
func (r *Request) FormFile(key string) (multipart.File, *multipart.FileHeader, error) {
	if r.MultipartForm == multipartByReader {
		return nil, nil, errors.New("http: multipart handled by MultipartReader")
	}
	if r.MultipartForm == nil {
		err := r.ParseMultipartForm(defaultMaxMemory)
		if err != nil {
			return nil, nil, err
		}
	}
	if r.MultipartForm != nil && r.MultipartForm.File != nil {
		if fhs := r.MultipartForm.File[key]; len(fhs) > 0 {
			f, err := fhs[0].Open()
			return f, fhs[0], err
		}
	}
	return nil, nil, ErrMissingFile
}

func (r *Request) expectsContinue() bool {
	return strings.ToLower(r.Header.Get("Expect")) == "100-continue"
}

func (r *Request) wantsHttp10KeepAlive() bool {
	if r.ProtoMajor != 1 || r.ProtoMinor != 0 {
		return false
	}
	return strings.Contains(strings.ToLower(r.Header.Get("Connection")), "keep-alive")
}
