// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Primitive HTTP client. See RFC 2616.

package http

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
)

// Given a string of the form "host", "host:port", or "[ipv6::address]:port",
// return true if the string includes a port.
func hasPort(s string) bool { return strings.LastIndex(s, ":") > strings.LastIndex(s, "]") }

// Used in Send to implement io.ReadCloser by bundling together the
// bufio.Reader through which we read the response, and the underlying
// network connection.
type readClose struct {
	io.Reader
	io.Closer
}

// matchNoProxy returns true if requests to addr should not use a proxy,
// according to the NO_PROXY or no_proxy environment variable.
func matchNoProxy(addr string) bool {
	if len(addr) == 0 {
		return false
	}
	no_proxy := os.Getenv("NO_PROXY")
	if len(no_proxy) == 0 {
		no_proxy = os.Getenv("no_proxy")
	}
	if no_proxy == "*" {
		return true
	}

	addr = strings.ToLower(strings.TrimSpace(addr))
	if hasPort(addr) {
		addr = addr[:strings.LastIndex(addr, ":")]
	}

	for _, p := range strings.Split(no_proxy, ",", -1) {
		p = strings.ToLower(strings.TrimSpace(p))
		if len(p) == 0 {
			continue
		}
		if hasPort(p) {
			p = p[:strings.LastIndex(p, ":")]
		}
		if addr == p || (p[0] == '.' && (strings.HasSuffix(addr, p) || addr == p[1:])) {
			return true
		}
	}
	return false
}

// Send issues an HTTP request.  Caller should close resp.Body when done reading it.
//
// TODO: support persistent connections (multiple requests on a single connection).
// send() method is nonpublic because, when we refactor the code for persistent
// connections, it may no longer make sense to have a method with this signature.
func send(req *Request) (resp *Response, err os.Error) {
	if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
		return nil, &badStringError{"unsupported protocol scheme", req.URL.Scheme}
	}

	addr := req.URL.Host
	if !hasPort(addr) {
		addr += ":" + req.URL.Scheme
	}
	info := req.URL.RawUserinfo
	if len(info) > 0 {
		enc := base64.URLEncoding
		encoded := make([]byte, enc.EncodedLen(len(info)))
		enc.Encode(encoded, []byte(info))
		if req.Header == nil {
			req.Header = make(Header)
		}
		req.Header.Set("Authorization", "Basic "+string(encoded))
	}

	var proxyURL *URL
	proxyAuth := ""
	proxy := ""
	if !matchNoProxy(addr) {
		proxy = os.Getenv("HTTP_PROXY")
		if proxy == "" {
			proxy = os.Getenv("http_proxy")
		}
	}

	if proxy != "" {
		proxyURL, err = ParseRequestURL(proxy)
		if err != nil {
			return nil, os.ErrorString("invalid proxy address")
		}
		if proxyURL.Host == "" {
			proxyURL, err = ParseRequestURL("http://" + proxy)
			if err != nil {
				return nil, os.ErrorString("invalid proxy address")
			}
		}
		addr = proxyURL.Host
		proxyInfo := proxyURL.RawUserinfo
		if proxyInfo != "" {
			enc := base64.URLEncoding
			encoded := make([]byte, enc.EncodedLen(len(proxyInfo)))
			enc.Encode(encoded, []byte(proxyInfo))
			proxyAuth = "Basic " + string(encoded)
		}
	}

	// Connect to server or proxy.
	conn, err := net.Dial("tcp", "", addr)
	if err != nil {
		return nil, err
	}

	if req.URL.Scheme == "http" {
		// Include proxy http header if needed.
		if proxyAuth != "" {
			req.Header.Set("Proxy-Authorization", proxyAuth)
		}
	} else { // https
		if proxyURL != nil {
			// Ask proxy for direct connection to server.
			// addr defaults above to ":https" but we need to use numbers
			addr = req.URL.Host
			if !hasPort(addr) {
				addr += ":443"
			}
			fmt.Fprintf(conn, "CONNECT %s HTTP/1.1\r\n", addr)
			fmt.Fprintf(conn, "Host: %s\r\n", addr)
			if proxyAuth != "" {
				fmt.Fprintf(conn, "Proxy-Authorization: %s\r\n", proxyAuth)
			}
			fmt.Fprintf(conn, "\r\n")

			// Read response.
			// Okay to use and discard buffered reader here, because
			// TLS server will not speak until spoken to.
			br := bufio.NewReader(conn)
			resp, err := ReadResponse(br, "CONNECT")
			if err != nil {
				return nil, err
			}
			if resp.StatusCode != 200 {
				f := strings.Split(resp.Status, " ", 2)
				return nil, os.ErrorString(f[1])
			}
		}

		// Initiate TLS and check remote host name against certificate.
		conn = tls.Client(conn, nil)
		if err = conn.(*tls.Conn).Handshake(); err != nil {
			return nil, err
		}
		h := req.URL.Host
		if hasPort(h) {
			h = h[:strings.LastIndex(h, ":")]
		}
		if err = conn.(*tls.Conn).VerifyHostname(h); err != nil {
			return nil, err
		}
	}

	err = req.Write(conn)
	if err != nil {
		conn.Close()
		return nil, err
	}

	reader := bufio.NewReader(conn)
	resp, err = ReadResponse(reader, req.Method)
	if err != nil {
		conn.Close()
		return nil, err
	}

	resp.Body = readClose{resp.Body, conn}

	return
}

// True if the specified HTTP status code is one for which the Get utility should
// automatically redirect.
func shouldRedirect(statusCode int) bool {
	switch statusCode {
	case StatusMovedPermanently, StatusFound, StatusSeeOther, StatusTemporaryRedirect:
		return true
	}
	return false
}

// Get issues a GET to the specified URL.  If the response is one of the following
// redirect codes, it follows the redirect, up to a maximum of 10 redirects:
//
//    301 (Moved Permanently)
//    302 (Found)
//    303 (See Other)
//    307 (Temporary Redirect)
//
// finalURL is the URL from which the response was fetched -- identical to the
// input URL unless redirects were followed.
//
// Caller should close r.Body when done reading it.
func Get(url string) (r *Response, finalURL string, err os.Error) {
	// TODO: if/when we add cookie support, the redirected request shouldn't
	// necessarily supply the same cookies as the original.
	// TODO: set referrer header on redirects.
	var base *URL
	for redirect := 0; ; redirect++ {
		if redirect >= 10 {
			err = os.ErrorString("stopped after 10 redirects")
			break
		}

		var req Request
		if base == nil {
			req.URL, err = ParseURL(url)
		} else {
			req.URL, err = base.ParseURL(url)
		}
		if err != nil {
			break
		}
		url = req.URL.String()
		if r, err = send(&req); err != nil {
			break
		}
		if shouldRedirect(r.StatusCode) {
			r.Body.Close()
			if url = r.Header.Get("Location"); url == "" {
				err = os.ErrorString(fmt.Sprintf("%d response missing Location header", r.StatusCode))
				break
			}
			base = req.URL
			continue
		}
		finalURL = url
		return
	}

	err = &URLError{"Get", url, err}
	return
}

// Post issues a POST to the specified URL.
//
// Caller should close r.Body when done reading it.
func Post(url string, bodyType string, body io.Reader) (r *Response, err os.Error) {
	var req Request
	req.Method = "POST"
	req.ProtoMajor = 1
	req.ProtoMinor = 1
	req.Close = true
	req.Body = nopCloser{body}
	req.Header = Header{
		"Content-Type": {bodyType},
	}
	req.TransferEncoding = []string{"chunked"}

	req.URL, err = ParseURL(url)
	if err != nil {
		return nil, err
	}

	return send(&req)
}

// PostForm issues a POST to the specified URL, 
// with data's keys and values urlencoded as the request body.
//
// Caller should close r.Body when done reading it.
func PostForm(url string, data map[string]string) (r *Response, err os.Error) {
	var req Request
	req.Method = "POST"
	req.ProtoMajor = 1
	req.ProtoMinor = 1
	req.Close = true
	body := urlencode(data)
	req.Body = nopCloser{body}
	req.Header = Header{
		"Content-Type":   {"application/x-www-form-urlencoded"},
		"Content-Length": {strconv.Itoa(body.Len())},
	}
	req.ContentLength = int64(body.Len())

	req.URL, err = ParseURL(url)
	if err != nil {
		return nil, err
	}

	return send(&req)
}

// TODO: remove this function when PostForm takes a multimap.
func urlencode(data map[string]string) (b *bytes.Buffer) {
	m := make(map[string][]string, len(data))
	for k, v := range data {
		m[k] = []string{v}
	}
	return bytes.NewBuffer([]byte(EncodeQuery(m)))
}

// Head issues a HEAD to the specified URL.
func Head(url string) (r *Response, err os.Error) {
	var req Request
	req.Method = "HEAD"
	if req.URL, err = ParseURL(url); err != nil {
		return
	}
	return send(&req)
}

type nopCloser struct {
	io.Reader
}

func (nopCloser) Close() os.Error { return nil }
