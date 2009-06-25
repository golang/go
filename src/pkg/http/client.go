// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Primitive HTTP client. See RFC 2616.

package http

import (
	"bufio";
	"fmt";
	"http";
	"io";
	"log";
	"net";
	"os";
	"strconv";
	"strings";
)

// Response represents the response from an HTTP request.
type Response struct {
	Status string; // e.g. "200 OK"
	StatusCode int; // e.g. 200

	// Header maps header keys to values.  If the response had multiple
	// headers with the same key, they will be concatenated, with comma
	// delimiters.  (Section 4.2 of RFC 2616 requires that multiple headers
	// be semantically equivalent to a comma-delimited sequence.)
	//
	// Keys in the map are canonicalized (see CanonicalHeaderKey).
	Header map [string] string;

	// Stream from which the response body can be read.
	Body io.ReadCloser;
}

// GetHeader returns the value of the response header with the given
// key, and true.  If there were multiple headers with this key, their
// values are concatenated, with a comma delimiter.  If there were no
// response headers with the given key, it returns the empty string and
// false.  Keys are not case sensitive.
func (r *Response) GetHeader(key string) (value string) {
	value, present := r.Header[CanonicalHeaderKey(key)];
	return;
}

// AddHeader adds a value under the given key.  Keys are not case sensitive.
func (r *Response) AddHeader(key, value string) {
	key = CanonicalHeaderKey(key);

	oldValues, oldValuesPresent := r.Header[key];
	if oldValuesPresent {
		r.Header[key] = oldValues + "," + value;
	} else {
		r.Header[key] = value;
	}
}

// Given a string of the form "host", "host:port", or "[ipv6::address]:port",
// return true if the string includes a port.
func hasPort(s string) bool {
	return strings.LastIndex(s, ":") > strings.LastIndex(s, "]");
}

// Used in Send to implement io.ReadCloser by bundling together the
// io.BufReader through which we read the response, and the underlying
// network connection.
type readClose struct {
	io.Reader;
	io.Closer;
}

// Send issues an HTTP request.  Caller should close resp.Body when done reading it.
//
// TODO: support persistent connections (multiple requests on a single connection).
// send() method is nonpublic because, when we refactor the code for persistent
// connections, it may no longer make sense to have a method with this signature.
func send(req *Request) (resp *Response, err os.Error) {
	if req.Url.Scheme != "http" {
		return nil, os.ErrorString("Unsupported protocol: " + req.Url.Scheme);
	}

	addr := req.Url.Host;
	if !hasPort(addr) {
		addr += ":http";
	}
	conn, err := net.Dial("tcp", "", addr);
	if err != nil {
		return nil, os.ErrorString("Error dialing " + addr + ": " + err.String());
	}

	// Close the connection if we encounter an error during header parsing.  We'll
	// cancel this when we hand the connection off to our caller.
	defer func() { if conn != nil { conn.Close() } }();

	err = req.write(conn);
	if err != nil {
		return nil, err;
	}

	// Parse the first line of the response.
	resp = new(Response);
	resp.Header = make(map[string] string);
	reader := bufio.NewReader(conn);

	line, err := readLine(reader);
	if err != nil {
		return nil, err;
	}
	f := strings.Split(line, " ", 3);
	if len(f) < 3 {
		return nil, os.ErrorString(fmt.Sprintf("Invalid first line in HTTP response: %q", line));
	}
	resp.Status = f[1] + " " + f[2];
	resp.StatusCode, err = strconv.Atoi(f[1]);
	if err != nil {
		return nil, os.ErrorString(fmt.Sprintf("Invalid status code in HTTP response: %q", line));
	}

	// Parse the response headers.
	for {
		key, value, err := readKeyValue(reader);
		if err != nil {
			return nil, err;
		}
		if key == "" {
			break; // end of response header
		}
		resp.AddHeader(key, value);
	}

	// TODO(rsc): Make this work:
	//   r := io.Reader(reader);
	var r io.Reader = reader;
	if v := resp.GetHeader("Transfer-Encoding"); v == "chunked" {
		r = newChunkedReader(reader);
	}
	resp.Body = readClose{ r, conn };

	conn = nil; // so that defered func won't close it
	err = nil;
	return;
}

// True if the specified HTTP status code is one for which the Get utility should
// automatically redirect.
func shouldRedirect(statusCode int) bool {
	switch statusCode {
	case StatusMovedPermanently, StatusFound, StatusSeeOther, StatusTemporaryRedirect:
		return true;
	}
	return false;
}

// Get issues a GET to the specified URL.  If the response is one of the following
// redirect codes, it follows the redirect, up to a maximum of 10 redirects:
//
//    301 (Moved Permanently)
//    302 (Found)
//    303 (See Other)
//    307 (Temporary Redirect)
//
// finalUrl is the URL from which the response was fetched -- identical to the input
// URL unless redirects were followed.
//
// Caller should close r.Body when done reading it.
func Get(url string) (r *Response, finalUrl string, err os.Error) {
	// TODO: if/when we add cookie support, the redirected request shouldn't
	// necessarily supply the same cookies as the original.
	// TODO: adjust referrer header on redirects.
	for redirectCount := 0; redirectCount < 10; redirectCount++ {
		var req Request;
		req.Url, err = ParseURL(url);
		if err != nil {
			return nil, url, err;
		}

		r, err := send(&req);
		if err != nil {
			return nil, url, err;
		}

		if !shouldRedirect(r.StatusCode) {
			return r, url, nil;
		}

		r.Body.Close();
		url := r.GetHeader("Location");
		if url == "" {
			return r, url, os.ErrorString("302 result with no Location header");
		}
	}

	return nil, url, os.ErrorString("Too many redirects");
}


// Post issues a POST to the specified URL.
//
// Caller should close r.Body when done reading it.
func Post(url string, bodyType string, body io.Reader) (r *Response, err os.Error) {
	var req Request;
	req.Method = "POST";
	req.Body = body;
	req.Header = map[string] string{
		"Content-Type": bodyType,
		"Transfer-Encoding": "chunked",
	};

	req.Url, err = ParseURL(url);
	if err != nil {
		return nil, err;
	}

	return send(&req);
}
