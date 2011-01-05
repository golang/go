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

var respExcludeHeader = map[string]bool{
	"Content-Length":    true,
	"Transfer-Encoding": true,
	"Trailer":           true,
}

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
	if len(f) < 2 {
		return nil, &badStringError{"malformed HTTP response", line}
	}
	reasonPhrase := ""
	if len(f) > 2 {
		reasonPhrase = f[2]
	}
	resp.Status = f[1] + " " + reasonPhrase
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

	err = readTransfer(resp, r)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// RFC2616: Should treat
//	Pragma: no-cache
// like
//	Cache-Control: no-cache
func fixPragmaCacheControl(header map[string]string) {
	if header["Pragma"] == "no-cache" {
		if _, presentcc := header["Cache-Control"]; !presentcc {
			header["Cache-Control"] = "no-cache"
		}
	}
}

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

// GetHeader returns the value of the response header with the given key.
// If there were multiple headers with this key, their values are concatenated,
// with a comma delimiter.  If there were no response headers with the given
// key, GetHeader returns an empty string.  Keys are not case sensitive.
func (r *Response) GetHeader(key string) (value string) {
	return r.Header[CanonicalHeaderKey(key)]
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
	text := resp.Status
	if text == "" {
		var ok bool
		text, ok = statusText[resp.StatusCode]
		if !ok {
			text = "status code " + strconv.Itoa(resp.StatusCode)
		}
	}
	io.WriteString(w, "HTTP/"+strconv.Itoa(resp.ProtoMajor)+".")
	io.WriteString(w, strconv.Itoa(resp.ProtoMinor)+" ")
	io.WriteString(w, strconv.Itoa(resp.StatusCode)+" "+text+"\r\n")

	// Process Body,ContentLength,Close,Trailer
	tw, err := newTransferWriter(resp)
	if err != nil {
		return err
	}
	err = tw.WriteHeader(w)
	if err != nil {
		return err
	}

	// Rest of header
	err = writeSortedKeyValue(w, resp.Header, respExcludeHeader)
	if err != nil {
		return err
	}

	// End-of-header
	io.WriteString(w, "\r\n")

	// Write body and trailer
	err = tw.WriteBody(w)
	if err != nil {
		return err
	}

	// Success
	return nil
}

func writeSortedKeyValue(w io.Writer, kvm map[string]string, exclude map[string]bool) os.Error {
	kva := make([]string, len(kvm))
	i := 0
	for k, v := range kvm {
		if !exclude[k] {
			kva[i] = fmt.Sprint(k + ": " + v + "\r\n")
			i++
		}
	}
	kva = kva[0:i]
	sort.SortStrings(kva)
	for _, l := range kva {
		if _, err := io.WriteString(w, l); err != nil {
			return err
		}
	}
	return nil
}
