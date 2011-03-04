// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
)

// DefaultTransport is the default implementation of Transport and is
// used by DefaultClient.  It establishes a new network connection for
// each call to Do and uses HTTP proxies as directed by the
// $HTTP_PROXY and $NO_PROXY (or $http_proxy and $no_proxy)
// environment variables.
var DefaultTransport Transport = &transport{}

// transport implements Tranport for the default case, using TCP
// connections to either the host or a proxy, serving http or https
// schemes.  In the future this may become public and support options
// on keep-alive connection duration, pipelining controls, etc.  For
// now this is simply a port of the old Go code client code to the
// Transport interface.
type transport struct {
	// TODO: keep-alives, pipelining, etc using a map from
	// scheme/host to a connection.  Something like:
	l        sync.Mutex
	hostConn map[string]*ClientConn
}

func (ct *transport) Do(req *Request) (resp *Response, err os.Error) {
	if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
		return nil, &badStringError{"unsupported protocol scheme", req.URL.Scheme}
	}

	addr := req.URL.Host
	if !hasPort(addr) {
		addr += ":" + req.URL.Scheme
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

	// Connect to server or proxy
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
