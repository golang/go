// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package synctest_test

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"testing/synctest"
	"time"
)

// Keep the following tests in sync with the package documentation.

func TestTime(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		start := time.Now() // always midnight UTC 2000-01-01
		go func() {
			time.Sleep(1 * time.Nanosecond)
			t.Log(time.Since(start)) // always logs "1ns"
		}()
		time.Sleep(2 * time.Nanosecond) // the AfterFunc will run before this Sleep returns
		t.Log(time.Since(start))        // always logs "2ns"
	})
}

func TestWait(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		done := false
		go func() {
			done = true
		}()
		// Wait will block until the goroutine above has finished.
		synctest.Wait()
		t.Log(done) // always logs "true"
	})
}

func TestContextAfterFunc(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// Create a context.Context which can be canceled.
		ctx, cancel := context.WithCancel(t.Context())

		// context.AfterFunc registers a function to be called
		// when a context is canceled.
		afterFuncCalled := false
		context.AfterFunc(ctx, func() {
			afterFuncCalled = true
		})

		// The context has not been canceled, so the AfterFunc is not called.
		synctest.Wait()
		if afterFuncCalled {
			t.Fatalf("before context is canceled: AfterFunc called")
		}

		// Cancel the context and wait for the AfterFunc to finish executing.
		// Verify that the AfterFunc ran.
		cancel()
		synctest.Wait()
		if !afterFuncCalled {
			t.Fatalf("after context is canceled: AfterFunc not called")
		}
	})
}

func TestContextWithTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// Create a context.Context which is canceled after a timeout.
		const timeout = 5 * time.Second
		ctx, cancel := context.WithTimeout(t.Context(), timeout)
		defer cancel()

		// Wait just less than the timeout.
		time.Sleep(timeout - time.Nanosecond)
		synctest.Wait()
		if err := ctx.Err(); err != nil {
			t.Fatalf("before timeout: ctx.Err() = %v, want nil\n", err)
		}

		// Wait the rest of the way until the timeout.
		time.Sleep(time.Nanosecond)
		synctest.Wait()
		if err := ctx.Err(); err != context.DeadlineExceeded {
			t.Fatalf("after timeout: ctx.Err() = %v, want DeadlineExceeded\n", err)
		}
	})
}

func TestHTTPTransport100Continue(t *testing.T) {
	synctest.Test(t, func(*testing.T) {
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
			req, _ := http.NewRequest("PUT", "http://test.tld/", strings.NewReader(body))
			req.Header.Set("Expect", "100-continue")
			resp, err := tr.RoundTrip(req)
			if err != nil {
				t.Errorf("RoundTrip: unexpected error %v\n", err)
			} else {
				resp.Body.Close()
			}
		}()

		// Read the request headers sent by the client.
		req, err := http.ReadRequest(bufio.NewReader(srvConn))
		if err != nil {
			t.Fatalf("ReadRequest: %v\n", err)
		}

		// Start a new goroutine copying the body sent by the client into a buffer.
		// Wait for all goroutines in the bubble to block and verify that we haven't
		// read anything from the client yet.
		var gotBody bytes.Buffer
		go io.Copy(&gotBody, req.Body)
		synctest.Wait()
		if got, want := gotBody.String(), ""; got != want {
			t.Fatalf("before sending 100 Continue, read body: %q, want %q\n", got, want)
		}

		// Write a "100 Continue" response to the client and verify that
		// it sends the request body.
		srvConn.Write([]byte("HTTP/1.1 100 Continue\r\n\r\n"))
		synctest.Wait()
		if got, want := gotBody.String(), body; got != want {
			t.Fatalf("after sending 100 Continue, read body: %q, want %q\n", got, want)
		}

		// Finish up by sending the "200 OK" response to conclude the request.
		srvConn.Write([]byte("HTTP/1.1 200 OK\r\n\r\n"))

		// We started several goroutines during the test.
		// The synctest.Test call will wait for all of them to exit before returning.
	})
}
