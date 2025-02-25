// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.synctest

package synctest_test

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing/synctest"
	"time"
)

// This example demonstrates testing [http.Transport]'s 100 Continue handling.
//
// An HTTP client sending a request can include an "Expect: 100-continue" header
// to tell the server that the client has additional data to send.
// The server may then respond with an 100 Continue information response
// to request the data, or some other status to tell the client the data is not needed.
// For example, a client uploading a large file might use this feature to confirm
// that the server is willing to accept the file before sending it.
//
// This test confirms that when sending an "Expect: 100-continue" header
// the HTTP client does not send a request's content before the server requests it,
// and that it does send the content after receiving a 100 Continue response.
func Example_httpTransport100Continue() {
	synctest.Run(func() {
		// Create an in-process fake network connection.
		// We cannot use a loopback network connection for this test,
		// because goroutines blocked on network I/O prevent a synctest
		// bubble from becoming idle.
		srvConn, cliConn := net.Pipe()
		defer cliConn.Close()
		defer srvConn.Close()

		tr := &http.Transport{
			// Use the fake network connection created above.
			DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
				return cliConn, nil
			},
			// Enable "Expect: 100-continue" handling.
			ExpectContinueTimeout: 5 * time.Second,
		}

		// Send a request with the "Expect: 100-continue" header set.
		// Send it in a new goroutine, since it won't complete until the end of the test.
		body := "request body"
		go func() {
			req, err := http.NewRequest("PUT", "http://test.tld/", strings.NewReader(body))
			if err != nil {
				panic(err)
			}
			req.Header.Set("Expect", "100-continue")
			resp, err := tr.RoundTrip(req)
			if err != nil {
				fmt.Printf("RoundTrip: unexpected error %v\n", err)
			} else {
				resp.Body.Close()
			}
		}()

		// Read the request headers sent by the client.
		req, err := http.ReadRequest(bufio.NewReader(srvConn))
		if err != nil {
			fmt.Printf("ReadRequest: %v\n", err)
			return
		}

		// Start a new goroutine copying the body sent by the client into a buffer.
		// Wait for all goroutines in the bubble to block and verify that we haven't
		// read anything from the client yet.
		var gotBody bytes.Buffer
		go io.Copy(&gotBody, req.Body)
		synctest.Wait()
		fmt.Printf("before sending 100 Continue, read body: %q\n", gotBody.String())

		// Write a "100 Continue" response to the client and verify that
		// it sends the request body.
		srvConn.Write([]byte("HTTP/1.1 100 Continue\r\n\r\n"))
		synctest.Wait()
		fmt.Printf("after sending 100 Continue, read body: %q\n", gotBody.String())

		// Finish up by sending the "200 OK" response to conclude the request.
		srvConn.Write([]byte("HTTP/1.1 200 OK\r\n\r\n"))

		// We started several goroutines during the test.
		// The synctest.Run call will wait for all of them to exit before returning.
	})

	// Output:
	// before sending 100 Continue, read body: ""
	// after sending 100 Continue, read body: "request body"
}
