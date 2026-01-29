// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
)

func TestTransportNewClientConnRoundTrip(t *testing.T) { run(t, testTransportNewClientConnRoundTrip) }
func testTransportNewClientConnRoundTrip(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		io.WriteString(w, req.Host)
	}), optFakeNet)

	scheme := mode.Scheme() // http or https
	cc, err := cst.tr.NewClientConn(t.Context(), scheme, cst.ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer cc.Close()

	// Send requests for a couple different domains.
	// All use the same connection.
	for _, host := range []string{"example.tld", "go.dev"} {
		req, _ := http.NewRequest("GET", fmt.Sprintf("%v://%v/", scheme, host), nil)
		resp, err := cc.RoundTrip(req)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := io.ReadAll(resp.Body)
		if string(got) != host {
			t.Errorf("got response body %q, want %v", got, host)
		}
		resp.Body.Close()

		// CloseIdleConnections does not close connections created by NewClientConn.
		cst.tr.CloseIdleConnections()
	}

	if err := cc.Err(); err != nil {
		t.Errorf("before close: ClientConn.Err() = %v, want nil", err)
	}

	cc.Close()
	if err := cc.Err(); err == nil {
		t.Errorf("after close: ClientConn.Err() = nil, want error")
	}

	req, _ := http.NewRequest("GET", scheme+"://example.tld/", nil)
	resp, err := cc.RoundTrip(req)
	if err == nil {
		resp.Body.Close()
		t.Errorf("after close: cc.RoundTrip succeeded, want error")
	}
	t.Log(err)
}

func newClientConnTest(t testing.TB, mode testMode, h http.HandlerFunc, opts ...any) (*clientServerTest, *http.ClientConn) {
	if h == nil {
		h = func(w http.ResponseWriter, req *http.Request) {}
	}
	cst := newClientServerTest(t, mode, h, opts...)
	cc, err := cst.tr.NewClientConn(t.Context(), mode.Scheme(), cst.ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cc.Close()
	})
	synctest.Wait()
	return cst, cc
}

// TestClientConnReserveAll reserves every concurrency slot on a connection.
func TestClientConnReserveAll(t *testing.T) { runSynctest(t, testClientConnReserveAll) }
func testClientConnReserveAll(t *testing.T, mode testMode) {
	cst, cc := newClientConnTest(t, mode, nil, optFakeNet, func(s *http.Server) {
		s.HTTP2 = &http.HTTP2Config{
			MaxConcurrentStreams: 3,
		}
	})

	want := 1
	switch mode {
	case http2Mode, http2UnencryptedMode:
		want = cst.ts.Config.HTTP2.MaxConcurrentStreams
	}
	available := cc.Available()
	if available != want {
		t.Fatalf("cc.Available() = %v, want %v", available, want)
	}

	// Reserve every available concurrency slot on the connection.
	for i := range available {
		if err := cc.Reserve(); err != nil {
			t.Fatalf("cc.Reserve() #%v = %v, want nil", i, err)
		}
		if got, want := cc.Available(), available-i-1; got != want {
			t.Fatalf("cc.Available() = %v, want %v", got, want)
		}
		if got, want := cc.InFlight(), i+1; got != want {
			t.Fatalf("cc.InFlight() = %v, want %v", got, want)
		}
	}

	// The next reservation attempt should fail, since every slot is consumed.
	if err := cc.Reserve(); err == nil {
		t.Fatalf("cc.Reserve() = nil, want error")
	}
}

// TestClientConnReserveParallel starts concurrent goroutines which reserve every
// concurrency slot on a connection.
func TestClientConnReserveParallel(t *testing.T) { runSynctest(t, testClientConnReserveParallel) }
func testClientConnReserveParallel(t *testing.T, mode testMode) {
	_, cc := newClientConnTest(t, mode, nil, optFakeNet, func(s *http.Server) {
		s.HTTP2 = &http.HTTP2Config{
			MaxConcurrentStreams: 3,
		}
	})
	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		success int
		failure int
	)
	available := cc.Available()
	const extra = 2
	for range available + extra {
		wg.Go(func() {
			err := cc.Reserve()
			mu.Lock()
			defer mu.Unlock()
			if err == nil {
				success++
			} else {
				failure++
			}
		})
	}
	wg.Wait()

	if got, want := success, available; got != want {
		t.Errorf("%v successful reservations, want %v", got, want)
	}
	if got, want := failure, extra; got != want {
		t.Errorf("%v failed reservations, want %v", got, want)
	}
}

// TestClientConnReserveRelease repeatedly reserves and releases concurrency slots.
func TestClientConnReserveRelease(t *testing.T) { runSynctest(t, testClientConnReserveRelease) }
func testClientConnReserveRelease(t *testing.T, mode testMode) {
	_, cc := newClientConnTest(t, mode, nil, optFakeNet, func(s *http.Server) {
		s.HTTP2 = &http.HTTP2Config{
			MaxConcurrentStreams: 3,
		}
	})

	available := cc.Available()
	for i := range 2 * available {
		if err := cc.Reserve(); err != nil {
			t.Fatalf("cc.Reserve() #%v = %v, want nil", i, err)
		}
		cc.Release()
	}

	if got, want := cc.Available(), available; got != want {
		t.Fatalf("cc.Available() = %v, want %v", available, want)
	}
}

// TestClientConnReserveAndConsume reserves a concurrency slot on a connection,
// and then verifies that various events consume the reservation.
func TestClientConnReserveAndConsume(t *testing.T) {
	for _, test := range []struct {
		name     string
		consume  func(t *testing.T, cc *http.ClientConn, mode testMode)
		handler  func(w http.ResponseWriter, req *http.Request, donec chan struct{})
		h1Closed bool
	}{{
		// Explicit release.
		name: "release",
		consume: func(t *testing.T, cc *http.ClientConn, mode testMode) {
			cc.Release()
		},
	}, {
		// Invalid request sent to RoundTrip.
		name: "invalid field name",
		consume: func(t *testing.T, cc *http.ClientConn, mode testMode) {
			req, _ := http.NewRequest("GET", mode.Scheme()+"://example.tld/", nil)
			req.Header["invalid field name"] = []string{"x"}
			_, err := cc.RoundTrip(req)
			if err == nil {
				t.Fatalf("RoundTrip succeeded, want failure")
			}
		},
	}, {
		// Successful request/response cycle.
		name: "body close",
		consume: func(t *testing.T, cc *http.ClientConn, mode testMode) {
			req, _ := http.NewRequest("GET", mode.Scheme()+"://example.tld/", nil)
			resp, err := cc.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip: %v", err)
			}
			resp.Body.Close()
		},
	}, {
		// Request context canceled before headers received.
		name: "cancel",
		consume: func(t *testing.T, cc *http.ClientConn, mode testMode) {
			ctx, cancel := context.WithCancel(t.Context())
			go func() {
				req, _ := http.NewRequestWithContext(ctx, "GET", mode.Scheme()+"://example.tld/", nil)
				_, err := cc.RoundTrip(req)
				if err == nil {
					t.Errorf("RoundTrip succeeded, want failure")
				}
			}()
			synctest.Wait()
			cancel()
		},
		handler: func(w http.ResponseWriter, req *http.Request, donec chan struct{}) {
			<-donec
		},
		// An HTTP/1 connection is closed after a request is canceled on it.
		h1Closed: true,
	}, {
		// Response body closed before full response received.
		name: "early body close",
		consume: func(t *testing.T, cc *http.ClientConn, mode testMode) {
			req, _ := http.NewRequest("GET", mode.Scheme()+"://example.tld/", nil)
			resp, err := cc.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip: %v", err)
			}
			t.Logf("%T", resp.Body)
			resp.Body.Close()
		},
		handler: func(w http.ResponseWriter, req *http.Request, donec chan struct{}) {
			w.WriteHeader(200)
			http.NewResponseController(w).Flush()
			<-donec
		},
		// An HTTP/1 connection is closed after a request is canceled on it.
		h1Closed: true,
	}} {
		t.Run(test.name, func(t *testing.T) {
			runSynctest(t, func(t *testing.T, mode testMode) {
				donec := make(chan struct{})
				defer close(donec)
				handler := func(w http.ResponseWriter, req *http.Request) {
					if test.handler != nil {
						test.handler(w, req, donec)
					}
				}

				_, cc := newClientConnTest(t, mode, handler, optFakeNet)
				stateHookCalls := 0
				cc.SetStateHook(func(cc *http.ClientConn) {
					stateHookCalls++
				})
				synctest.Wait()
				stateHookCalls = 0 // ignore any initial update call

				avail := cc.Available()
				if err := cc.Reserve(); err != nil {
					t.Fatalf("cc.Reserve() = %v, want nil", err)
				}
				synctest.Wait()
				if got, want := stateHookCalls, 0; got != want {
					t.Errorf("connection state hook calls: %v, want %v", got, want)
				}

				test.consume(t, cc, mode)
				synctest.Wait()

				// State hook should be called, either to report the
				// connection availability increasing or the connection closing,
				// or both.
				if stateHookCalls == 0 {
					t.Errorf("connection state hook calls: %v, want >1", stateHookCalls)
				}

				if test.h1Closed && (mode == http1Mode || mode == https1Mode) {
					if got, want := cc.Available(), 0; got != want {
						t.Errorf("cc.Available() = %v, want %v", got, want)
					}
					if got, want := cc.InFlight(), 0; got != want {
						t.Errorf("cc.InFlight() = %v, want %v", got, want)
					}
					if err := cc.Err(); err == nil {
						t.Errorf("cc.Err() = nil, want closed connection")
					}
				} else {
					if got, want := cc.Available(), avail; got != want {
						t.Errorf("cc.Available() = %v, want %v", got, want)
					}
					if got, want := cc.InFlight(), 0; got != want {
						t.Errorf("cc.InFlight() = %v, want %v", got, want)
					}
					if err := cc.Err(); err != nil {
						t.Errorf("cc.Err() = %v, want nil", err)
					}
				}

				if cc.Available() > 0 {
					if err := cc.Reserve(); err != nil {
						t.Errorf("cc.Reserve() = %v, want success", err)
					}
				}
			})
		})
	}

}

// TestClientConnRoundTripBlocks verifies that RoundTrip blocks until a concurrency
// slot is available on a connection.
func TestClientConnRoundTripBlocks(t *testing.T) { runSynctest(t, testClientConnRoundTripBlocks) }
func testClientConnRoundTripBlocks(t *testing.T, mode testMode) {
	var handlerCalls atomic.Int64
	requestc := make(chan struct{})
	handler := func(w http.ResponseWriter, req *http.Request) {
		handlerCalls.Add(1)
		<-requestc
	}
	_, cc := newClientConnTest(t, mode, handler, optFakeNet, func(s *http.Server) {
		s.HTTP2 = &http.HTTP2Config{
			MaxConcurrentStreams: 3,
		}
	})

	available := cc.Available()
	var responses atomic.Int64
	const extra = 2
	for range available + extra {
		go func() {
			req, _ := http.NewRequest("GET", mode.Scheme()+"://example.tld/", nil)
			resp, err := cc.RoundTrip(req)
			responses.Add(1)
			if err != nil {
				t.Errorf("RoundTrip: %v", err)
				return
			}
			resp.Body.Close()
		}()
	}

	synctest.Wait()
	if got, want := int(handlerCalls.Load()), available; got != want {
		t.Errorf("got %v handler calls, want %v", got, want)
	}
	if got, want := int(responses.Load()), 0; got != want {
		t.Errorf("got %v responses, want %v", got, want)
	}

	for i := range available + extra {
		requestc <- struct{}{}
		synctest.Wait()
		if got, want := int(responses.Load()), i+1; got != want {
			t.Errorf("got %v responses, want %v", got, want)
		}
	}
}
