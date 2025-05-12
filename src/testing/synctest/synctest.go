// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package synctest provides support for testing concurrent code.
//
// The [Test] function runs a function in an isolated "bubble".
// Any goroutines started within the bubble are also part of the bubble.
//
// # Time
//
// Within a bubble, the [time] package uses a fake clock.
// Each bubble has its own clock.
// The initial time is midnight UTC 2000-01-01.
//
// Time in a bubble only advances when every goroutine in the
// bubble is durably blocked.
// See below for the exact definition of "durably blocked".
//
// For example, this test runs immediately rather than taking
// two seconds:
//
//	func TestTime(t *testing.T) {
//		synctest.Test(t, func(t *testing.T) {
//			start := time.Now() // always midnight UTC 2001-01-01
//			go func() {
//				time.Sleep(1 * time.Nanosecond)
//				t.Log(time.Since(start)) // always logs "1ns"
//			}()
//			time.Sleep(2 * time.Nanosecond) // the goroutine above will run before this Sleep returns
//			t.Log(time.Since(start))        // always logs "2ns"
//		})
//	}
//
// Time stops advancing when the root goroutine of the bubble exits.
//
// # Blocking
//
// A goroutine in a bubble is "durably blocked" when it is blocked
// and can only be unblocked by another goroutine in the same bubble.
// A goroutine which can be unblocked by an event from outside its
// bubble is not durably blocked.
//
// The [Wait] function blocks until all other goroutines in the
// bubble are durably blocked.
//
// For example:
//
//	func TestWait(t *testing.T) {
//		synctest.Test(t, func(t *testing.T) {
//			done := false
//			go func() {
//				done = true
//			}()
//			// Wait will block until the goroutine above has finished.
//			synctest.Wait()
//			t.Log(done) // always logs "true"
//		})
//	}
//
// When every goroutine in a bubble is durably blocked:
//
//   - [Wait] returns, if it has been called.
//   - Otherwise, time advances to the next time that will
//     unblock at least one goroutine, if there is such a time
//     and the root goroutine of the bubble has not exited.
//   - Otherwise, there is a deadlock and [Test] panics.
//
// The following operations durably block a goroutine:
//
//   - a blocking send or receive on a channel created within the bubble
//   - a blocking select statement where every case is a channel created
//     within the bubble
//   - [sync.Cond.Wait]
//   - [sync.WaitGroup.Wait]
//   - [time.Sleep]
//
// Locking a [sync.Mutex] or [sync.RWMutex] is not durably blocking.
//
// # Isolation
//
// A channel, [time.Timer], or [time.Ticker] created within a bubble
// is associated with it. Operating on a bubbled channel, timer, or
// ticker from outside the bubble panics.
//
// # Example: Context.AfterFunc
//
// This example demonstrates testing the [context.AfterFunc] function.
//
// AfterFunc registers a function to execute in a new goroutine
// after a context is canceled.
//
// The test verifies that the function is not run before the context is canceled,
// and is run after the context is canceled.
//
//	func TestContextAfterFunc(t *testing.T) {
//		synctest.Test(t, func(t *testing.T) {
//			// Create a context.Context which can be canceled.
//			ctx, cancel := context.WithCancel(t.Context())
//
//			// context.AfterFunc registers a function to be called
//			// when a context is canceled.
//			afterFuncCalled := false
//			context.AfterFunc(ctx, func() {
//				afterFuncCalled = true
//			})
//
//			// The context has not been canceled, so the AfterFunc is not called.
//			synctest.Wait()
//			if afterFuncCalled {
//				t.Fatalf("before context is canceled: AfterFunc called")
//			}
//
//			// Cancel the context and wait for the AfterFunc to finish executing.
//			// Verify that the AfterFunc ran.
//			cancel()
//			synctest.Wait()
//			if !afterFuncCalled {
//				t.Fatalf("before context is canceled: AfterFunc not called")
//			}
//		})
//	}
//
// # Example: Context.WithTimeout
//
// This example demonstrates testing the [context.WithTimeout] function.
//
// WithTimeout creates a context which is canceled after a timeout.
//
// The test verifies that the context is not canceled before the timeout expires,
// and is canceled after the timeout expires.
//
//	func TestContextWithTimeout(t *testing.T) {
//		synctest.Test(t, func(t *testing.T) {
//			// Create a context.Context which is canceled after a timeout.
//			const timeout = 5 * time.Second
//			ctx, cancel := context.WithTimeout(t.Context(), timeout)
//			defer cancel()
//
//			// Wait just less than the timeout.
//			time.Sleep(timeout - time.Nanosecond)
//			synctest.Wait()
//			if err := ctx.Err(); err != nil {
//				t.Fatalf("before timeout: ctx.Err() = %v, want nil\n", err)
//			}
//
//			// Wait the rest of the way until the timeout.
//			time.Sleep(time.Nanosecond)
//			synctest.Wait()
//			if err := ctx.Err(); err != context.DeadlineExceeded {
//				t.Fatalf("after timeout: ctx.Err() = %v, want DeadlineExceeded\n", err)
//			}
//		})
//	}
//
// # Example: HTTP 100 Continue
//
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
//
//	func TestHTTPTransport100Continue(t *testing.T) {
//		synctest.Test(t, func(*testing.T) {
//			// Create an in-process fake network connection.
//			// We cannot use a loopback network connection for this test,
//			// because goroutines blocked on network I/O prevent a synctest
//			// bubble from becoming idle.
//			srvConn, cliConn := net.Pipe()
//			defer cliConn.Close()
//			defer srvConn.Close()
//
//			tr := &http.Transport{
//				// Use the fake network connection created above.
//				DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
//					return cliConn, nil
//				},
//				// Enable "Expect: 100-continue" handling.
//				ExpectContinueTimeout: 5 * time.Second,
//			}
//
//			// Send a request with the "Expect: 100-continue" header set.
//			// Send it in a new goroutine, since it won't complete until the end of the test.
//			body := "request body"
//			go func() {
//				req, _ := http.NewRequest("PUT", "http://test.tld/", strings.NewReader(body))
//				req.Header.Set("Expect", "100-continue")
//				resp, err := tr.RoundTrip(req)
//				if err != nil {
//					t.Errorf("RoundTrip: unexpected error %v\n", err)
//				} else {
//					resp.Body.Close()
//				}
//			}()
//
//			// Read the request headers sent by the client.
//			req, err := http.ReadRequest(bufio.NewReader(srvConn))
//			if err != nil {
//				t.Fatalf("ReadRequest: %v\n", err)
//			}
//
//			// Start a new goroutine copying the body sent by the client into a buffer.
//			// Wait for all goroutines in the bubble to block and verify that we haven't
//			// read anything from the client yet.
//			var gotBody bytes.Buffer
//			go io.Copy(&gotBody, req.Body)
//			synctest.Wait()
//			if got, want := gotBody.String(), ""; got != want {
//				t.Fatalf("before sending 100 Continue, read body: %q, want %q\n", got, want)
//			}
//
//			// Write a "100 Continue" response to the client and verify that
//			// it sends the request body.
//			srvConn.Write([]byte("HTTP/1.1 100 Continue\r\n\r\n"))
//			synctest.Wait()
//			if got, want := gotBody.String(), body; got != want {
//				t.Fatalf("after sending 100 Continue, read body: %q, want %q\n", got, want)
//			}
//
//			// Finish up by sending the "200 OK" response to conclude the request.
//			srvConn.Write([]byte("HTTP/1.1 200 OK\r\n\r\n"))
//
//			// We started several goroutines during the test.
//			// The synctest.Test call will wait for all of them to exit before returning.
//		})
//	}
package synctest

import (
	"internal/synctest"
	"testing"
	_ "unsafe" // for linkname
)

// Test executes f in a new bubble.
//
// Test waits for all goroutines in the bubble to exit before returning.
// If the goroutines in the bubble become deadlocked, the test fails.
//
// Test must not be called from within a bubble.
//
// The [*testing.T] provided to f has the following properties:
//
//   - T.Cleanup functions run inside the bubble,
//     immediately before Test returns.
//   - T.Context returns a [context.Context] with a Done channel
//     associated with the bubble.
//   - T.Run, T.Parallel, and T.Deadline must not be called.
func Test(t *testing.T, f func(*testing.T)) {
	synctest.Run(func() {
		testingSynctestTest(t, f)
	})
}

//go:linkname testingSynctestTest testing/synctest.testingSynctestTest
func testingSynctestTest(t *testing.T, f func(*testing.T))

// Wait blocks until every goroutine within the current bubble,
// other than the current goroutine, is durably blocked.
//
// Wait must not be called from outside a bubble.
// Wait must not be called concurrently by multiple goroutines
// in the same bubble.
func Wait() {
	synctest.Wait()
}
