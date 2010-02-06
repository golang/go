// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP Response reading and parsing.

package http

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

var respExcludeHeader = map[string]int{}

// Response represents the response from an HTTP request.
//
type Response struct {
	Status     string // e.g. "200 OK"
	StatusCode int    // e.g. 200
	Proto      string // e.g. "HTTP/1.0"
	ProtoMajor int    // e.g. 1
	ProtoMinor int    // e.g. 0

	// RequestMethod records the method used in the HTTP request.
	// Header fields such as Content-Length have method-specific meaning.
	RequestMethod string // e.g. "HEAD", "CONNECT", "GET", etc.

	// Header maps header keys to values.  If the response had multiple
	// headers with the same key, they will be concatenated, with comma
	// delimiters.  (Section 4.2 of RFC 2616 requires that multiple headers
	// be semantically equivalent to a comma-delimited sequence.) Values
	// duplicated by other fields in this struct (e.g., ContentLength) are
	// omitted from Header.
	//
	// Keys in the map are canonicalized (see CanonicalHeaderKey).
	Header map[string]string

	// Body represents the response body.
	Body io.ReadCloser

	// ContentLength records the length of the associated content.  The
	// value -1 indicates that the length is unknown.  Unless RequestMethod
	// is "HEAD", values >= 0 indicate that the given number of bytes may
	// be read from Body.
	ContentLength int64

	// Contains transfer encodings from outer-most to inner-most. Value is
	// nil, means that "identity" encoding is used.
	TransferEncoding []string

	// Close records whether the header directed that the connection be
	// closed after reading Body.  The value is advice for clients: neither
	// ReadResponse nor Response.Write ever closes a connection.
	Close bool

	// Trailer maps trailer keys to values.  Like for Header, if the
	// response has multiple trailer lines with the same key, they will be
	// concatenated, delimited by commas.
	Trailer map[string]string
}

// ReadResponse reads and returns an HTTP response from r.  The RequestMethod
// parameter specifies the method used in the corresponding request (e.g.,
// "GET", "HEAD").  Clients must call resp.Body.Close when finished reading
// resp.Body.  After that call, clients can inspect resp.Trailer to find
// key/value pairs included in the response trailer.
func ReadResponse(r *bufio.Reader, requestMethod string) (resp *Response, err os.Error) {

	resp = new(Response)

	resp.RequestMethod = strings.ToUpper(requestMethod)

	// Parse the first line of the response.
	line, err := readLine(r)
	if err != nil {
		return nil, err
	}
	f := strings.Split(line, " ", 3)
	if len(f) < 3 {
		return nil, &badStringError{"malformed HTTP response", line}
	}
	resp.Status = f[1] + " " + f[2]
	resp.StatusCode, err = strconv.Atoi(f[1])
	if err != nil {
		return nil, &badStringError{"malformed HTTP status code", f[1]}
	}

	resp.Proto = f[0]
	var ok bool
	if resp.ProtoMajor, resp.ProtoMinor, ok = parseHTTPVersion(resp.Proto); !ok {
		return nil, &badStringError{"malformed HTTP version", resp.Proto}
	}

	// Parse the response headers.
	nheader := 0
	resp.Header = make(map[string]string)
	for {
		key, value, err := readKeyValue(r)
		if err != nil {
			return nil, err
		}
		if key == "" {
			break // end of response header
		}
		if nheader++; nheader >= maxHeaderLines {
			return nil, ErrHeaderTooLong
		}
		resp.AddHeader(key, value)
	}

	fixPragmaCacheControl(resp.Header)

	// Transfer encoding, content length
	resp.TransferEncoding, err = fixTransferEncoding(resp.Header)
	if err != nil {
		return nil, err
	}

	resp.ContentLength, err = fixLength(resp.StatusCode, resp.RequestMethod,
		resp.Header, resp.TransferEncoding)
	if err != nil {
		return nil, err
	}

	// Closing
	resp.Close = shouldClose(resp.ProtoMajor, resp.ProtoMinor, resp.Header)

	// Trailer
	resp.Trailer, err = fixTrailer(resp.Header, resp.TransferEncoding)
	if err != nil {
		return nil, err
	}

	// Prepare body reader.  ContentLength < 0 means chunked encoding
	// or close connection when finished, since multipart is not supported yet
	switch {
	case chunked(resp.TransferEncoding):
		resp.Body = &body{Reader: newChunkedReader(r), hdr: resp, r: r, closing: resp.Close}
	case resp.ContentLength >= 0:
		resp.Body = &body{Reader: io.LimitReader(r, resp.ContentLength), closing: resp.Close}
	default:
		resp.Body = &body{Reader: r, closing: resp.Close}
	}

	return resp, nil
}

// body turns a Reader into a ReadCloser.
// Close ensures that the body has been fully read
// and then reads the trailer if necessary.
type body struct {
	io.Reader
	hdr     interface{}   // non-nil (Response or Request) value means read trailer
	r       *bufio.Reader // underlying wire-format reader for the trailer
	closing bool          // is the connection to be closed after reading body?
}

func (b *body) Close() os.Error {
	if b.hdr == nil && b.closing {
		// no trailer and closing the connection next.
		// no point in reading to EOF.
		return nil
	}

	trashBuf := make([]byte, 1024) // local for thread safety
	for {
		_, err := b.Read(trashBuf)
		if err == nil {
			continue
		}
		if err == os.EOF {
			break
		}
		return err
	}
	if b.hdr == nil { // not reading trailer
		return nil
	}

	// TODO(petar): Put trailer reader code here

	return nil
}

// RFC2616: Should treat
//	Pragma: no-cache
// like
//	Cache-Control: no-cache
func fixPragmaCacheControl(header map[string]string) {
	if v, present := header["Pragma"]; present && v == "no-cache" {
		if _, presentcc := header["Cache-Control"]; !presentcc {
			header["Cache-Control"] = "no-cache"
		}
	}
}

// Parse the trailer header
func fixTrailer(header map[string]string, te []string) (map[string]string, os.Error) {
	raw, present := header["Trailer"]
	if !present {
		return nil, nil
	}

	header["Trailer"] = "", false
	trailer := make(map[string]string)
	keys := strings.Split(raw, ",", 0)
	for _, key := range keys {
		key = CanonicalHeaderKey(strings.TrimSpace(key))
		switch key {
		case "Transfer-Encoding", "Trailer", "Content-Length":
			return nil, &badStringError{"bad trailer key", key}
		}
		trailer[key] = ""
	}
	if len(trailer) == 0 {
		return nil, nil
	}
	if !chunked(te) {
		// Trailer and no chunking
		return nil, ErrUnexpectedTrailer
	}
	return trailer, nil
}

// Sanitize transfer encoding
func fixTransferEncoding(header map[string]string) ([]string, os.Error) {
	raw, present := header["Transfer-Encoding"]
	if !present {
		return nil, nil
	}

	header["Transfer-Encoding"] = "", false
	encodings := strings.Split(raw, ",", 0)
	te := make([]string, 0, len(encodings))
	// TODO: Even though we only support "identity" and "chunked"
	// encodings, the loop below is designed with foresight. One
	// invariant that must be maintained is that, if present,
	// chunked encoding must always come first.
	for _, encoding := range encodings {
		encoding = strings.ToLower(strings.TrimSpace(encoding))
		// "identity" encoding is not recored
		if encoding == "identity" {
			break
		}
		if encoding != "chunked" {
			return nil, &badStringError{"unsupported transfer encoding", encoding}
		}
		te = te[0 : len(te)+1]
		te[len(te)-1] = encoding
	}
	if len(te) > 1 {
		return nil, &badStringError{"too many transfer encodings", strings.Join(te, ",")}
	}
	if len(te) > 0 {
		// Chunked encoding trumps Content-Length. See RFC 2616
		// Section 4.4. Currently len(te) > 0 implies chunked
		// encoding.
		header["Content-Length"] = "", false
		return te, nil
	}

	return nil, nil
}

func noBodyExpected(requestMethod string) bool {
	return requestMethod == "HEAD"
}

// Determine the expected body length, using RFC 2616 Section 4.4. This
// function is not a method, because ultimately it should be shared by
// ReadResponse and ReadRequest.
func fixLength(status int, requestMethod string, header map[string]string, te []string) (int64, os.Error) {

	// Logic based on response type or status
	if noBodyExpected(requestMethod) {
		return 0, nil
	}
	if status/100 == 1 {
		return 0, nil
	}
	switch status {
	case 204, 304:
		return 0, nil
	}

	// Logic based on Transfer-Encoding
	if chunked(te) {
		return -1, nil
	}

	// Logic based on Content-Length
	if cl, present := header["Content-Length"]; present {
		cl = strings.TrimSpace(cl)
		if cl != "" {
			n, err := strconv.Atoi64(cl)
			if err != nil || n < 0 {
				return -1, &badStringError{"bad Content-Length", cl}
			}
			return n, nil
		} else {
			header["Content-Length"] = "", false
		}
	}

	// Logic based on media type. The purpose of the following code is just
	// to detect whether the unsupported "multipart/byteranges" is being
	// used. A proper Content-Type parser is needed in the future.
	if ct, present := header["Content-Type"]; present {
		ct = strings.ToLower(ct)
		if strings.Index(ct, "multipart/byteranges") >= 0 {
			return -1, ErrNotSupported
		}
	}


	// Logic based on close
	return -1, nil
}

// Determine whether to hang up after sending a request and body, or
// receiving a response and body
func shouldClose(major, minor int, header map[string]string) bool {
	if major < 1 || (major == 1 && minor < 1) {
		return true
	} else if v, present := header["Connection"]; present {
		// TODO: Should split on commas, toss surrounding white space,
		// and check each field.
		if v == "close" {
			return true
		}
	}
	return false
}

// Checks whether chunked is part of the encodings stack
func chunked(te []string) bool { return len(te) > 0 && te[0] == "chunked" }

// AddHeader adds a value under the given key.  Keys are not case sensitive.
func (r *Response) AddHeader(key, value string) {
	key = CanonicalHeaderKey(key)

	oldValues, oldValuesPresent := r.Header[key]
	if oldValuesPresent {
		r.Header[key] = oldValues + "," + value
	} else {
		r.Header[key] = value
	}
}

// GetHeader returns the value of the response header with the given
// key, and true.  If there were multiple headers with this key, their
// values are concatenated, with a comma delimiter.  If there were no
// response headers with the given key, it returns the empty string and
// false.  Keys are not case sensitive.
func (r *Response) GetHeader(key string) (value string) {
	value, _ = r.Header[CanonicalHeaderKey(key)]
	return
}

// ProtoAtLeast returns whether the HTTP protocol used
// in the response is at least major.minor.
func (r *Response) ProtoAtLeast(major, minor int) bool {
	return r.ProtoMajor > major ||
		r.ProtoMajor == major && r.ProtoMinor >= minor
}

// Writes the response (header, body and trailer) in wire format. This method
// consults the following fields of resp:
//
//  StatusCode
//  ProtoMajor
//  ProtoMinor
//  RequestMethod
//  TransferEncoding
//  Trailer
//  Body
//  ContentLength
//  Header, values for non-canonical keys will have unpredictable behavior
//
func (resp *Response) Write(w io.Writer) os.Error {

	// RequestMethod should be upper-case
	resp.RequestMethod = strings.ToUpper(resp.RequestMethod)

	// Status line
	text, ok := statusText[resp.StatusCode]
	if !ok {
		text = "status code " + strconv.Itoa(resp.StatusCode)
	}
	io.WriteString(w, "HTTP/"+strconv.Itoa(resp.ProtoMajor)+".")
	io.WriteString(w, strconv.Itoa(resp.ProtoMinor)+" ")
	io.WriteString(w, strconv.Itoa(resp.StatusCode)+" "+text+"\r\n")

	// Sanitize the field triple (Body, ContentLength, TransferEncoding)
	if noBodyExpected(resp.RequestMethod) {
		resp.Body = nil
		resp.TransferEncoding = nil
		// resp.ContentLength is expected to hold Content-Length
		if resp.ContentLength < 0 {
			return ErrMissingContentLength
		}
	} else {
		if !resp.ProtoAtLeast(1, 1) || resp.Body == nil {
			resp.TransferEncoding = nil
		}
		if chunked(resp.TransferEncoding) {
			resp.ContentLength = -1
		} else if resp.Body == nil { // no chunking, no body
			resp.ContentLength = 0
		}
	}

	// Write Content-Length and/or Transfer-Encoding whose values are a
	// function of the sanitized field triple (Body, ContentLength,
	// TransferEncoding)
	if chunked(resp.TransferEncoding) {
		io.WriteString(w, "Transfer-Encoding: chunked\r\n")
	} else {
		if resp.ContentLength > 0 || resp.RequestMethod == "HEAD" {
			io.WriteString(w, "Content-Length: ")
			io.WriteString(w, strconv.Itoa64(resp.ContentLength)+"\r\n")
		}
	}
	if resp.Header != nil {
		resp.Header["Content-Length"] = "", false
		resp.Header["Transfer-Encoding"] = "", false
	}

	// Sanitize Trailer
	if !chunked(resp.TransferEncoding) {
		resp.Trailer = nil
	} else if resp.Trailer != nil {
		// TODO: At some point, there should be a generic mechanism for
		// writing long headers, using HTTP line splitting
		io.WriteString(w, "Trailer: ")
		needComma := false
		for k, _ := range resp.Trailer {
			k = CanonicalHeaderKey(k)
			switch k {
			case "Transfer-Encoding", "Trailer", "Content-Length":
				return &badStringError{"invalid Trailer key", k}
			}
			if needComma {
				io.WriteString(w, ",")
			}
			io.WriteString(w, k)
			needComma = true
		}
		io.WriteString(w, "\r\n")
	}
	if resp.Header != nil {
		resp.Header["Trailer"] = "", false
	}

	// Rest of header
	writeSortedKeyValue(w, resp.Header, respExcludeHeader)

	// End-of-header
	io.WriteString(w, "\r\n")

	// Write body
	if resp.Body != nil {
		var err os.Error
		if chunked(resp.TransferEncoding) {
			cw := NewChunkedWriter(w)
			_, err = io.Copy(cw, resp.Body)
			if err == nil {
				err = cw.Close()
			}
		} else {
			_, err = io.Copy(w, io.LimitReader(resp.Body, resp.ContentLength))
		}
		if err != nil {
			return err
		}
		if err = resp.Body.Close(); err != nil {
			return err
		}
	}

	// TODO(petar): Place trailer writer code here.
	if chunked(resp.TransferEncoding) {
		// Last chunk, empty trailer
		io.WriteString(w, "\r\n")
	}

	// Success
	return nil
}

func writeSortedKeyValue(w io.Writer, kvm map[string]string, exclude map[string]int) {
	kva := make([]string, len(kvm))
	i := 0
	for k, v := range kvm {
		if _, exc := exclude[k]; !exc {
			kva[i] = fmt.Sprint(k + ": " + v + "\r\n")
			i++
		}
	}
	kva = kva[0:i]
	sort.SortStrings(kva)
	for _, l := range kva {
		io.WriteString(w, l)
	}
}
