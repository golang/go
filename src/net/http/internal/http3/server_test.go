// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"golang.org/x/net/quic"
)

// requestHeader is a helper function to make sure that all required
// pseudo-headers exist in an http.Header used for a request. Per
// https://www.rfc-editor.org/rfc/rfc9114.html#name-request-pseudo-header-field:
// "All HTTP/3 requests MUST include exactly one value for the :method,
// :scheme, and :path pseudo-header fields, unless the request is a CONNECT
// request;"
func requestHeader(h http.Header) http.Header {
	minimalHeader := http.Header{
		":method": {"GET"},
		":scheme": {"https"},
		":path":   {"/"},
	}
	maps.Copy(minimalHeader, h)
	return minimalHeader
}

func TestServerReceivePushStream(t *testing.T) {
	// "[...] if a server receives a client-initiated push stream,
	// this MUST be treated as a connection error of type H3_STREAM_CREATION_ERROR."
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-6.2.2-3
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, nil)
		tc := ts.connect()
		tc.newStream(streamTypePush)
		tc.wantClosed("invalid client-created push stream", errH3StreamCreationError)
	})
}

func TestServerCancelPushForUnsentPromise(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, nil)
		tc := ts.connect()
		tc.greet()

		const pushID = 100
		tc.control.writeVarint(int64(frameTypeCancelPush))
		tc.control.writeVarint(int64(sizeVarint(pushID)))
		tc.control.writeVarint(pushID)
		tc.control.Flush()

		tc.wantClosed("client canceled never-sent push ID", errH3IDError)
	})
}

func TestServerHeader(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			header := w.Header()
			for key, values := range r.Header {
				for _, value := range values {
					header.Add(key, value)
				}
			}
			w.WriteHeader(204)
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"header-from-client": {"that", "should", "be", "echoed"},
		}))
		reqStream.wantSomeHeaders(http.Header{
			":status":            {"204"},
			"Header-From-Client": {"that", "should", "be", "echoed"},
		})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHeaderSnapshot(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-Test-Header", "original")
			w.WriteHeader(200)
			w.Header().Set("X-Test-Header", "modified")
			w.Write([]byte("body"))
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{
			":status":       {"200"},
			"X-Test-Header": {"original"},
		})
		reqStream.wantData([]byte("body"))
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHeaderInvalid(t *testing.T) {
	tests := []struct {
		name      string
		header    http.Header
		wantError bool
	}{
		{
			name:      "header name with control character",
			header:    http.Header{"name\nevilinjection": {"Value"}},
			wantError: true,
		},
		{
			name:      "header name with uppercase character",
			header:    http.Header{"nAme": {"Value"}},
			wantError: true,
		},
		{
			name:      "pseudo-header name with control character",
			header:    http.Header{":path\nevilinjection": {"Value"}},
			wantError: true,
		},
		{
			name:      "pseudo-header name with uppercase character",
			header:    http.Header{":meThod": {"Value"}},
			wantError: true,
		},
		{
			name:      "header value with control character",
			header:    http.Header{"name": {"Value\nEvilInjection"}},
			wantError: true,
		},
		{
			name:      "pseudo-header value with control character",
			header:    http.Header{":method": {"Value\nEvilInjection"}},
			wantError: true,
		},
		{
			name:      "connection header name",
			header:    http.Header{"connection": {"foo"}},
			wantError: true,
		},
		{
			name:      "keep-alive header name",
			header:    http.Header{"Keep-Alive": {"foo"}},
			wantError: true,
		},
		{
			name:      "proxy-connection header name",
			header:    http.Header{"proxy-connection": {"foo"}},
			wantError: true,
		},
		{
			name:      "transfer-encoding header name",
			header:    http.Header{"transfer-encoding": {"foo"}},
			wantError: true,
		},
		{
			name:      "upgrade header name",
			header:    http.Header{"upgrade": {"foo"}},
			wantError: true,
		},
		{
			name:      "te header name",
			header:    http.Header{"te": {"foo"}},
			wantError: true,
		},
		{
			name:      "te header name with trailers value",
			header:    http.Header{"te": {"trailers"}},
			wantError: false,
		},
	}
	for _, tt := range tests {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			body := []byte("some data")
			ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write(body)
			}))
			tc := ts.connect()
			tc.greet()

			reqStream := tc.newStream(streamTypeRequest)
			reqStream.writeHeadersRaw(requestHeader(tt.header))

			if tt.wantError {
				reqStream.wantError(quic.StreamError(errH3MessageError))
			} else {
				reqStream.wantHeaders(nil)
				reqStream.wantData(body)
				reqStream.wantClosed("request is complete")
			}
		})
	}
}

func TestServerPseudoHeader(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Pseudo-headers from client request should populate a specific
			// field in http.Request, and should not be part of http.Request.Header.
			if len(r.Header) != 0 {
				t.Errorf("got %v, want request header to be empty", r.Header)
			}
			if r.Method != "GET" {
				t.Errorf("got %v, want GET method", r.Method)
			}
			if r.Host != "fake.tld:1234" {
				t.Errorf("got %v, want fake.tld:1234", r.Host)
			}
			wantURL := &url.URL{
				Path:     "/some/path",
				RawQuery: "query=value&query2=value2#fragment",
			}
			if !reflect.DeepEqual(r.URL, wantURL) {
				t.Errorf("got %v, want URL to be %v", r.URL, wantURL)
			}

			// Conversely, server should not be able to set pseudo-headers by
			// writing to the ResponseWriter's Header.
			header := w.Header()
			header.Add(":status", "123")
			w.WriteHeader(321)
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(http.Header{
			":method":    {"GET"},
			":authority": {"fake.tld:1234"},
			":scheme":    {"https"},
			":path":      {"/some/path?query=value&query2=value2#fragment"},
		})
		reqStream.wantSomeHeaders(http.Header{":status": {"321"}})
		reqStream.wantClosed("request is complete")

		reqStream = tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(http.Header{}) // Missing pseudo-header.
		reqStream.wantError(quic.StreamError(errH3MessageError))
	})
}

func TestServerPseudoHeaderCount(t *testing.T) {
	tests := []struct {
		name      string
		header    http.Header
		wantError bool
	}{
		{
			name: "missing method pseudo-header",
			header: http.Header{
				":scheme":    {"https"},
				":path":      {"/"},
				":authority": {"fake.tld:1234"},
			},
			wantError: true,
		},
		{
			name: "valid pseudo-headers for non-CONNECT request",
			header: http.Header{
				":method": {"GET"},
				":scheme": {"https"},
				":path":   {"/"},
			},
			wantError: false,
		},
		{
			name: "extraneous pseudo-headers for non-CONNECT request",
			header: http.Header{
				":method": {"GET", "GET"}, // Duplicate :method.
				":scheme": {"https"},
				":path":   {"/"},
			},
			wantError: true,
		},
		{
			name: "missing pseudo-headers for non-CONNECT request",
			header: http.Header{
				":method": {"GET", "GET"},
				":path":   {"/"},
			},
			wantError: true,
		},
		{
			name: "valid pseudo-headers for CONNECT request",
			header: http.Header{
				":method":    {"CONNECT"},
				":authority": {"fake.tld:1234"},
			},
			wantError: false,
		},
		{
			name: "extraneous pseudo-headers for CONNECT request",
			header: http.Header{
				":method":    {"CONNECT"},
				":authority": {"fake.tld:1234"},
				":path":      {"/"}, // :path should be omitted.
			},
			wantError: true,
		},
		{
			name: "missing pseudo-headers for CONNECT request",
			header: http.Header{
				":method": {"CONNECT"},
			},
			wantError: true,
		},
	}
	for _, tt := range tests {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			body := []byte("some data")
			ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write(body)
			}))
			tc := ts.connect()
			tc.greet()

			reqStream := tc.newStream(streamTypeRequest)
			reqStream.writeHeaders(tt.header)

			if tt.wantError {
				reqStream.wantError(quic.StreamError(errH3MessageError))
			} else {
				reqStream.wantHeaders(nil)
				reqStream.wantData(body)
				reqStream.wantClosed("request is complete")
			}
		})
	}
}

func TestServerInvalidHeader(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Add("valid-name", "valid value")
			// Invalid headers are skipped.
			w.Header().Add("invalid name with spaces", "some value")
			w.Header().Add("some-name", "invalid value with \n")
			w.Header().Add("valid-name-2", "valid value 2")
			w.WriteHeader(200)
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{
			":status":      {"200"},
			"Valid-Name":   {"valid value"},
			"Valid-Name-2": {"valid value 2"},
		})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerAuthorityAndHostHeader(t *testing.T) {
	for _, test := range []struct {
		name     string
		h        http.Header
		valid    bool
		wantHost string
	}{{
		name: "authority host mismatch",
		h: http.Header{
			":authority": {"example.tld"},
			"host":       {"other.tld"},
		},
	}, {
		// The RFCs aren't explicit on whether a :authority and host that
		// differ only in case is a mismatch. We treat it as a mismatch because
		// there doesn't seem to be a good reason not to.
		name: "authority host case differs",
		h: http.Header{
			":authority": {"example.tld"},
			"host":       {"EXAMPLE.TLD"},
		},
	}, {
		name: "authority and multiple host",
		h: http.Header{
			":authority": {"example.tld"},
			"host":       {"example.tld", "example.tld"},
		},
	}, {
		name: "multiple host only",
		h: http.Header{
			"host": {"example.tld", "example.tld"},
		},
	}, {
		name: "multiple authority only",
		h: http.Header{
			":authority": {"example.tld", "example.tld"},
		},
	}, {
		name: "empty authority",
		h: http.Header{
			":authority": {""},
		},
	}, {
		name: "invalid authority",
		h: http.Header{
			":authority": {"example . tld"},
		},
	}, {
		name: "invalid host",
		h: http.Header{
			"host": {"example . tld"},
		},
	}, {
		name: "authority only",
		h: http.Header{
			":authority": {"example.tld"},
		},
		valid:    true,
		wantHost: "example.tld",
	}, {
		name: "host only",
		h: http.Header{
			"host": {"example.tld"},
		},
		valid:    true,
		wantHost: "example.tld",
	}, {
		name: "authority host match",
		h: http.Header{
			":authority": {"example.tld"},
			"host":       {"example.tld"},
		},
		valid:    true,
		wantHost: "example.tld",
	}, {
		name: "authority host match with port",
		h: http.Header{
			":authority": {"example.tld:443"},
			"host":       {"example.tld:443"},
		},
		valid:    true,
		wantHost: "example.tld:443",
	}, {
		name: "authority host mismatch with port",
		h: http.Header{
			":authority": {"example.tld:80"},
			"host":       {"example.tld:443"},
		},
	}, {
		name: "userinfo in authority",
		h: http.Header{
			":authority": {"user:pass@example.tld"},
		},
	}, {
		name: "userinfo in host",
		h: http.Header{
			"host": {"user:pass@example.tld"},
		},
	}, {
		name:     "neither authority nor host",
		h:        http.Header{},
		valid:    true,
		wantHost: "",
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			ts := newTestServer(t, nil)
			tc := ts.connect()
			tc.greet()

			reqStream := tc.newStream(streamTypeRequest)
			reqStream.writeHeaders(requestHeader(test.h))
			if test.valid {
				call := tc.nextHandlerCall()
				if call == nil {
					t.Fatal("no server handler call; want one")
				}
				if got, want := call.req.Host, test.wantHost; got != want {
					t.Errorf("handler got Host %q, want %q", got, want)
				}
				if h, ok := call.req.Header["Host"]; ok {
					t.Errorf(`handler got Header["Host"] = %q, want unset`, h)
				}
			} else {
				reqStream.wantError(quic.StreamError(errH3MessageError))
			}
		})
	}
}

func TestServerInvalidStatus(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		gotpanic := make(chan bool)
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer close(gotpanic)
			defer func() {
				if e := recover(); e != nil {
					got := fmt.Sprintf("%T, %v", e, e)
					want := "string, invalid WriteHeader code 0"
					if got != want {
						t.Errorf("unexpected panic value:\n got: %v\nwant: %v\n", got, want)
					}
					gotpanic <- true
					// Set an explicit 503. This also tests that the
					// WriteHeader call panics before it recorded that an
					// explicit value was set.
					w.WriteHeader(503)

					// Verify that writing invalid status will not panic if a
					// status is already set anyways.
					w.WriteHeader(0)
				}
			}()
			w.WriteHeader(0) // Invalid. Will panic.
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		if !<-gotpanic {
			t.Error("expected panic in handler")
		}
		synctest.Wait()
		reqStream.wantSomeHeaders(http.Header{
			":status": {"503"},
		})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHeaderLimits(t *testing.T) {
	for _, test := range []struct {
		name                string
		h                   http.Header
		valid               bool
		maxHeaderBytes      int
		maxHeaderValueCount int
	}{{
		name: "within limits",
		h: http.Header{
			"x-foo": {strings.Repeat("x", 1000)},
		},
		maxHeaderBytes: 1500,
		valid:          true,
	}, {
		name: "too many header bytes",
		h: http.Header{
			"x-foo": {strings.Repeat("x", 1000)},
			"x-bar": {strings.Repeat("x", 1000)},
		},
		maxHeaderBytes: 1500,
	}, {
		name: "field count within limit",
		h: http.Header{
			// :method, :scheme, :path, plus:
			"x-foo": {"4"},
			"x-bar": {"5"},
		},
		maxHeaderBytes:      1500,
		maxHeaderValueCount: 5,
		valid:               true,
	}, {
		name: "field count over limit",
		h: http.Header{
			// :method, :scheme, :path, plus:
			"x-foo": {"4"},
			"x-bar": {"5"},
		},
		maxHeaderBytes:      1500,
		maxHeaderValueCount: 4,
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			if test.maxHeaderValueCount != 0 {
				t.Skip("TODO: when we support only go1.27")
			}
			ts := newTestServer(t, nil)
			ts.s.srv1.MaxHeaderBytes = test.maxHeaderBytes
			// TODO: When we only support go1.27.
			//ts.s.srv1.MaxHeaderValueCount = test.maxHeaderValueCount
			tc := ts.connect()
			tc.greet()

			reqStream := tc.newStream(streamTypeRequest)
			reqStream.writeHeaders(requestHeader(test.h))
			if test.valid {
				call := tc.nextHandlerCall()
				if call == nil {
					t.Fatal("no server handler call; want one")
				}
			} else {
				reqStream.wantError(quic.StreamError(errH3RequestRejected))
			}
		})
	}
}

func TestServerBody(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatal(err)
			}
			w.Write([]byte(r.URL.Path)) // Implicitly calls w.WriteHeader(200).
			w.Write(body)
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		bodyContent := []byte("some body content that should be echoed")
		reqStream.writeData(bodyContent)
		reqStream.CloseWrite()
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		// Small multiple calls to Write will be coalesced into one DATA frame.
		reqStream.wantData(append([]byte("/"), bodyContent...))
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHeadResponseNoBody(t *testing.T) {
	bodyContent := []byte("response body that will not be sent for HEAD requests")
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Write(bodyContent)
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData(bodyContent)
		reqStream.wantClosed("request is complete")

		reqStream = tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{":method": {http.MethodHead}}))
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerShutdownGoaway(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, nil)

		tc := ts.connect()
		tc.greet()
		tc.wantNotClosed("after initial connection handshake")

		requestCount := int64(5)
		for range requestCount {
			tc.newStream(streamTypeRequest).writeHeaders(requestHeader(nil))
		}

		control := tc.wantStream(streamTypeControl)
		control.wantSettings(nil)

		shutdownComplete := make(chan any)
		go func() {
			ts.s.shutdown(t.Context())
			shutdownComplete <- struct{}{}
		}()
		control.wantGoaway((requestCount - 1) * 4) // Request stream ID goes from 0, 4, 8, ...
		<-shutdownComplete
	})
}

func TestServerHandlerEmpty(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Empty handler should return a 200 OK
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHandlerFlushing(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(time.Second)
			w.Write([]byte("first"))

			time.Sleep(time.Second)
			w.Write([]byte("second"))
			w.(http.Flusher).Flush()

			time.Sleep(time.Second)
			w.Write([]byte("third"))
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		respBody := make([]byte, 100)

		time.Sleep(time.Second)
		synctest.Wait()
		if n, err := reqStream.Read(respBody); err == nil {
			t.Errorf("got %v bytes read, want no message yet", n)
		}

		time.Sleep(time.Second)
		synctest.Wait()
		if _, err := reqStream.Read(respBody); err != nil {
			t.Errorf("failed to read partial response from server, got err: %v", err)
		}

		time.Sleep(time.Second)
		synctest.Wait()
		if _, err := reqStream.Read(respBody); err != io.EOF {
			t.Errorf("got err %v, want EOF", err)
		}
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHandlerStreaming(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		stream := make(chan string)
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Flushing when we have not written anything yet implicitly calls
			// w.WriteHeader(200).
			w.(http.Flusher).Flush()
			for str := range stream {
				w.Write([]byte(str))
				w.(http.Flusher).Flush()
			}
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})

		for _, data := range []string{"a", "bunch", "of", "things", "to", "stream"} {
			stream <- data
			reqStream.wantData([]byte(data))
		}
		close(stream)
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHandlerDeclaresContentLength(t *testing.T) {
	tests := []struct {
		name             string
		contentLen       string
		actualContentLen int
		wantWrittenLen   int
		wantTrimmed      bool
		wantCLHeader     bool
	}{
		{
			name:             "accurate content length",
			contentLen:       "100",
			actualContentLen: 100,
			wantWrittenLen:   100,
			wantCLHeader:     true,
		},
		{
			name:             "larger content length",
			contentLen:       "100",
			actualContentLen: 10,
			wantWrittenLen:   10,
			wantCLHeader:     true,
		},
		{
			name:             "smaller content length",
			contentLen:       "10",
			actualContentLen: 100,
			wantWrittenLen:   10,
			wantTrimmed:      true,
			wantCLHeader:     true,
		},
		{
			name:             "non-numeric string",
			contentLen:       "intentional gibberish",
			actualContentLen: 100,
			wantWrittenLen:   100,
		},
		{
			name:             "negative number",
			contentLen:       "-10",
			actualContentLen: 100,
			wantWrittenLen:   100,
		},
		{
			name:             "plus sign",
			contentLen:       "+10",
			actualContentLen: 100,
			wantWrittenLen:   100,
		},
		{
			name:             "empty",
			contentLen:       "",
			actualContentLen: 100,
			wantWrittenLen:   100,
		},
		{
			name:             "large valid content length",
			contentLen:       "3000000000",
			actualContentLen: 100,
			wantWrittenLen:   100,
			wantCLHeader:     true,
		},
		{
			name:             "content length overflowing int64",
			contentLen:       "9223372036854775808",
			actualContentLen: 100,
			wantWrittenLen:   100,
		},
	}

	for _, tt := range tests {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Length", tt.contentLen)
				var written int
				var lastErr error
				for range tt.actualContentLen {
					n, err := w.Write([]byte("a"))
					written += n
					lastErr = err
				}
				if tt.wantTrimmed != (lastErr != nil) {
					t.Errorf("got %v error when writing response body, even though wantTrimmed is %v", lastErr, tt.wantTrimmed)
				}
				if written != tt.wantWrittenLen {
					t.Errorf("got %v bytes written by the server, want %v bytes", written, tt.wantWrittenLen)
				}
			}))
			tc := ts.connect()
			tc.greet()

			reqStream := tc.newStream(streamTypeRequest)
			reqStream.writeHeaders(requestHeader(nil))
			expectedHeaders := http.Header{
				":status":      {"200"},
				"Content-Type": {"text/plain; charset=utf-8"},
				"Date":         {"Sat, 01 Jan 2000 00:00:00 GMT"}, // Synctest starting time.
			}
			if tt.wantCLHeader {
				expectedHeaders.Set("Content-Length", tt.contentLen)
			}
			reqStream.wantHeaders(expectedHeaders)
			reqStream.wantData(slices.Repeat([]byte("a"), tt.wantWrittenLen))
			reqStream.wantClosed("request is complete")
		})
	}
}

func TestServerExpect100Continue(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		streamIdle := make(chan bool)
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Expect: 100-continue header should not be accessible from the
			// server handler.
			if len(r.Header) > 0 {
				t.Errorf("got %v, want request header to be empty", r.Header)
			}
			// Reading the body will cause the server to call w.WriteHeader(100).
			<-streamIdle
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatal(err)
			}
			// Implicitly calls w.WriteHeader(200) since non-1XX status code
			// has been sent yet so far.
			w.Write(body)
		}))
		tc := ts.connect()
		tc.greet()

		// Client sends an Expect: 100-continue request.
		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"expect": {"100-continue"},
		}))

		reqStream.wantIdle("stream is idle until server sends an HTTP 100 status")
		streamIdle <- true
		// Wait until server responds with HTTP status 100 before sending the
		// body.
		reqStream.wantSomeHeaders(http.Header{":status": {"100"}})
		body := []byte("body that will be echoed back if we get status 100")
		reqStream.writeData(body)
		reqStream.CloseWrite()

		// Receive the server's response after sending the body.
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData(body)
		reqStream.wantClosed("request is complete")
	})
}

func TestServerExpect100ContinueSentManually(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(100)
			body, err := io.ReadAll(r.Body) // Should not send another 100.
			if err != nil {
				t.Fatal(err)
			}
			w.Write(body)
		}))
		tc := ts.connect()
		tc.greet()

		// Client sends an Expect: 100-continue request.
		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"expect": {"100-continue"},
		}))

		// Send the body once the server responds with HTTP status 100.
		reqStream.wantSomeHeaders(http.Header{":status": {"100"}})
		body := []byte("body that will be echoed back")
		reqStream.writeData(body)
		reqStream.CloseWrite()

		// Verify that the server responds with 200, rather than another 100.
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData(body)
		reqStream.wantClosed("request is complete")
	})
}

func TestServerExpect100ContinueRejected(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rejectBody := []byte("not allowed")
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(403)
			w.Write(rejectBody)
		}))
		tc := ts.connect()
		tc.greet()

		// Client sends an Expect: 100-continue request.
		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"expect": {"100-continue"},
		}))

		// Server rejects it.
		reqStream.wantSomeHeaders(http.Header{":status": {"403"}})
		reqStream.wantData(rejectBody)
		reqStream.wantClosed("request is complete")
	})
}

func TestServer100ContinueBodyReadAfterFinalResponse(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(200)
			w.(http.Flusher).Flush()
			// Read should not cause an HTTP 100 status to be sent since we
			// have sent an HTTP 200 response already.
			// Read should also return an error and should not hang.
			if _, err := io.ReadAll(r.Body); err == nil {
				t.Errorf("got %v, want an error", err)
			}
		}))
		tc := ts.connect()
		tc.greet()

		// Client sends an Expect: 100-continue request.
		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"expect": {"100-continue"},
		}))

		// Verify that no HTTP 100 was sent.
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantClosed("request is complete")
	})
}

func TestServer100ContinueBodyReadAfter100AndFinalResponse(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		body := []byte("client body")
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(100)
			w.WriteHeader(200)
			w.(http.Flusher).Flush()
			// Allow Read to succeed since the handler has sent 100 prior to 200.
			if gotBody, err := io.ReadAll(r.Body); err != nil || string(gotBody) != string(body) {
				t.Errorf("io.ReadAll(r.Body) = %v, %v; want %v, nil", gotBody, err, body)
			}
		}))
		tc := ts.connect()
		tc.greet()

		// Client sends an Expect: 100-continue request.
		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"expect": {"100-continue"},
		}))

		// Send the body once the server responds with HTTP status 100.
		reqStream.wantSomeHeaders(http.Header{":status": {"100"}})
		reqStream.writeData(body)
		reqStream.CloseWrite()
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHandlerReadReqWithNoBody(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		serverBody := []byte("hello from server!")
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if _, err := io.ReadAll(r.Body); err != nil {
				t.Errorf("got %v err when reading from an empty request body, want nil", err)
			}
			w.Write(serverBody)
		}))
		tc := ts.connect()
		tc.greet()

		// Case 1: we know that there is no body / DATA frame because the
		// client closes the write direction of the stream.
		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.CloseWrite()
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData(serverBody)
		reqStream.wantClosed("request is complete")

		// Case 2: we know that there is no body / DATA frame because the
		// client indicates a Content-Length of 0.
		reqStream = tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"content-length": {"0"},
		}))
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData(serverBody)
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHandlerReadTrailer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		body := []byte("some body")
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			wantTrailer := http.Header{
				"Client-Trailer-A": nil,
				"Client-Trailer-B": nil,
			}
			if !reflect.DeepEqual(r.Trailer, wantTrailer) {
				t.Errorf("got %v; want trailer to be %v before reading the body", r.Trailer, wantTrailer)
			}
			if _, err := io.ReadAll(r.Body); err != nil {
				t.Fatal(err)
			}
			wantTrailer = http.Header{
				"Client-Trailer-A": {"valuea"},
				"Client-Trailer-B": {"valueb"},
			}
			if !reflect.DeepEqual(r.Trailer, wantTrailer) {
				t.Errorf("got %v; want trailer to be %v after reading the body", r.Trailer, wantTrailer)
			}
			w.WriteHeader(200)
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"trailer": {"Client-Trailer-A, Client-Trailer-B"},
		}))
		reqStream.writeData(body)
		reqStream.writeHeaders(http.Header{
			"Client-Trailer-A": {"valuea"},
			"Client-Trailer-B": {"valueb"},
			// Server should not accept undeclared trailers.
			"Undeclared-Trailer": {"undeclared"},
		})
		reqStream.wantHeaders(nil)
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHandlerReadTrailerNoBody(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			wantTrailer := http.Header{
				"Client-Trailer-A": nil,
				"Client-Trailer-B": nil,
			}
			if !reflect.DeepEqual(r.Trailer, wantTrailer) {
				t.Errorf("got %v; want trailer to be %v before reading the body", r.Trailer, wantTrailer)
			}
			if _, err := io.ReadAll(r.Body); err != nil {
				t.Fatal(err)
			}
			wantTrailer = http.Header{
				"Client-Trailer-A": {"valuea"},
				"Client-Trailer-B": {"valueb"},
			}
			if !reflect.DeepEqual(r.Trailer, wantTrailer) {
				t.Errorf("got %v; want trailer to be %v after reading the body", r.Trailer, wantTrailer)
			}
			w.WriteHeader(200)
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			"trailer":        {"Client-Trailer-A, Client-Trailer-B"},
			"content-length": {"0"},
		}))
		reqStream.writeHeaders(http.Header{
			"Client-Trailer-A": {"valuea"},
			"Client-Trailer-B": {"valueb"},
			// Server should not accept undeclared trailers.
			"Undeclared-Trailer": {"undeclared"},
		})
		reqStream.wantHeaders(nil)
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHandlerWriteTrailer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		body := []byte("some body")
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Trailer", "server-trailer-a, server-trailer-b") // Trailer header will be canonicalized.
			w.Header().Add("Trailer", "Server-Trailer-C")

			w.Write(body)

			w.Header().Set("server-trailer-a", "valuea") // Trailer header will be canonicalized.
			w.Header().Set("Server-Trailer-C", "valuec") // skipping B
			// Server should not send undeclared trailers, unless it has the
			// magic "Trailer:" prefix.
			w.Header().Set("Server-Trailer-Not-Declared", "should be omitted")
			w.Header().Set("Trailer:Undeclared-Trailer-Exception", "should be sent")
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{
			":status": {"200"},
			"Trailer": {"Server-Trailer-A, Server-Trailer-B, Server-Trailer-C"},
		})
		reqStream.wantData(body)
		reqStream.wantSomeHeaders(http.Header{
			"Server-Trailer-A":             {"valuea"},
			"Server-Trailer-C":             {"valuec"},
			"Undeclared-Trailer-Exception": {"should be sent"},
		})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerHandlerWriteTrailerNoBody(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Trailer", "server-trailer-a, server-trailer-b") // Trailer header will be canonicalized.
			w.Header().Add("Trailer", "Server-Trailer-C")

			w.(http.Flusher).Flush()

			w.Header().Set("server-trailer-a", "valuea") // Trailer header will be canonicalized.
			w.Header().Set("Server-Trailer-C", "valuec") // skipping B
			// Server should not send undeclared trailers without "Trailer:"
			// prefix.
			w.Header().Set("Server-Trailer-Not-Declared", "should be omitted")
			w.Header().Set("Trailer:undeclared-trailer-exception", "should be sent")
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{
			":status": {"200"},
			"Trailer": {"Server-Trailer-A, Server-Trailer-B, Server-Trailer-C"},
		})
		reqStream.wantSomeHeaders(http.Header{
			"Server-Trailer-A":             {"valuea"},
			"Server-Trailer-C":             {"valuec"},
			"Undeclared-Trailer-Exception": {"should be sent"},
		})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerInfersHeaders(t *testing.T) {
	tests := []struct {
		name            string
		flushedEarly    bool
		responseStatus  int
		does100Continue bool
		declaredHeader  http.Header
		want            http.Header
	}{
		{
			name:           "infers undeclared headers",
			responseStatus: 200,
			declaredHeader: http.Header{
				"Some-Other-Header": {"some value"},
			},
			want: http.Header{
				"Date":              {"Sat, 01 Jan 2000 00:00:00 GMT"}, // Synctest starting time.
				"Content-Type":      {"text/html; charset=utf-8"},
				"Some-Other-Header": {"some value"},
			},
		},
		{
			name:           "does not write over declared header",
			responseStatus: 200,
			declaredHeader: http.Header{
				"Date":              {"some date"},
				"Content-Type":      {"some content type"},
				"Some-Other-Header": {"some value"},
			},
			want: http.Header{
				"Date":              {"some date"},
				"Content-Type":      {"some content type"},
				"Some-Other-Header": {"some value"},
			},
		},
		{
			name:           "does not infer content type for response with no body",
			responseStatus: 304, // 304 status response has no body.
			declaredHeader: http.Header{
				"Some-Other-Header": {"some value"},
			},
			want: http.Header{
				"Date":              {"Sat, 01 Jan 2000 00:00:00 GMT"}, // Synctest starting time.
				"Some-Other-Header": {"some value"},
			},
		},
		{
			// See golang.org/issue/31753.
			name:           "does not infer content type for response with declared content encoding",
			responseStatus: 200,
			declaredHeader: http.Header{
				"Content-Encoding":  {"some encoding"},
				"Some-Other-Header": {"some value"},
			},
			want: http.Header{
				"Date":              {"Sat, 01 Jan 2000 00:00:00 GMT"}, // Synctest starting time.
				"Content-Encoding":  {"some encoding"},
				"Some-Other-Header": {"some value"},
			},
		},
		{
			name:           "infers content type for response with empty content encoding",
			responseStatus: 200,
			declaredHeader: http.Header{
				"Content-Encoding":  {""},
				"Some-Other-Header": {"some value"},
			},
			want: http.Header{
				"Date":              {"Sat, 01 Jan 2000 00:00:00 GMT"}, // Synctest starting time.
				"Content-Encoding":  {""},
				"Content-Type":      {"text/html; charset=utf-8"},
				"Some-Other-Header": {"some value"},
			},
		},
		{
			name:           "does not infer content type when header is flushed before body is written",
			responseStatus: 200,
			flushedEarly:   true,
			declaredHeader: http.Header{
				"Some-Other-Header": {"some value"},
			},
			want: http.Header{
				"Date":              {"Sat, 01 Jan 2000 00:00:00 GMT"}, // Synctest starting time.
				"Some-Other-Header": {"some value"},
			},
		},
		{
			name:            "infers header for the header that comes after 100 continue",
			responseStatus:  200,
			does100Continue: true,
			declaredHeader: http.Header{
				"Some-Other-Header": {"some value"},
			},
			want: http.Header{
				"Date":              {"Sat, 01 Jan 2000 00:00:00 GMT"}, // Synctest starting time.
				"Content-Type":      {"text/html; charset=utf-8"},
				"Some-Other-Header": {"some value"},
			},
		},
	}

	for _, tt := range tests {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			body := []byte("<html>some html content</html>")
			streamIdle := make(chan bool)
			ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.does100Continue {
					<-streamIdle
					io.ReadAll(r.Body)
				}
				for name, values := range tt.declaredHeader {
					for _, value := range values {
						w.Header().Add(name, value)
					}
				}
				w.WriteHeader(tt.responseStatus)
				if tt.flushedEarly {
					w.(http.Flusher).Flush()
				}
				// Write the body one byte at a time. To confirm that body
				// writes are buffered and that Content-Type will not be
				// wrongly identified as text/plain rather than text/html.
				for _, b := range body {
					w.Write([]byte{b})
				}
			}))
			tc := ts.connect()
			tc.greet()

			reqStream := tc.newStream(streamTypeRequest)

			if tt.does100Continue {
				reqStream.writeHeaders(requestHeader(http.Header{
					"expect": {"100-continue"},
				}))
				reqStream.wantIdle("stream is idle until server sends an HTTP 100 status")
				streamIdle <- true
				reqStream.wantHeaders(http.Header{":status": {"100"}})
			}

			reqStream.writeHeaders(requestHeader(nil))
			tt.want.Add(":status", strconv.Itoa(tt.responseStatus))
			reqStream.wantHeaders(tt.want)
			if responseCanHaveBody(tt.responseStatus) {
				reqStream.wantData(body)
			}
			reqStream.wantClosed("request is complete")
		})
	}
}

func TestServerBuffersBodyWrite(t *testing.T) {
	tests := []struct {
		name      string
		bodyLen   int
		writeSize int
		flushes   bool
	}{
		{
			name:      "buffers small body content",
			bodyLen:   defaultBodyBufferCap * 10,
			writeSize: 5,
			flushes:   false,
		},
		{
			name:      "does not buffer large body content",
			bodyLen:   defaultBodyBufferCap * 10,
			writeSize: defaultBodyBufferCap * 2,
			flushes:   false,
		},
		{
			name:      "does not buffer flushed body content",
			bodyLen:   defaultBodyBufferCap * 10,
			writeSize: 10,
			flushes:   true,
		},
	}
	for _, tt := range tests {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				for n := 0; n < tt.bodyLen; n += tt.writeSize {
					data := slices.Repeat([]byte("a"), min(tt.writeSize, tt.bodyLen-n))
					n, err := w.Write(data)
					if err != nil {
						t.Fatal(err)
					}
					if n != len(data) {
						t.Errorf("got %v bytes when writing in server handler, want %v", n, len(data))
					}
					if tt.flushes {
						w.(http.Flusher).Flush()
					}
				}
			}))
			tc := ts.connect()
			tc.greet()

			reqStream := tc.newStream(streamTypeRequest)
			reqStream.writeHeaders(requestHeader(nil))
			reqStream.wantHeaders(nil)
			switch {
			case tt.writeSize > defaultBodyBufferCap:
				// After using the buffer once, it is no longer used since the
				// writeSize is larger than the buffer.
				for n := 0; n < tt.bodyLen; n += tt.writeSize {
					reqStream.wantData(slices.Repeat([]byte("a"), min(tt.writeSize, tt.bodyLen-n)))
				}
			case tt.flushes:
				for n := 0; n < tt.bodyLen; n += tt.writeSize {
					reqStream.wantData(slices.Repeat([]byte("a"), min(tt.writeSize, tt.bodyLen-n)))
				}
			case tt.writeSize <= defaultBodyBufferCap:
				dataLen := defaultBodyBufferCap + tt.writeSize - (defaultBodyBufferCap % tt.writeSize)
				for n := 0; n < tt.bodyLen; n += dataLen {
					reqStream.wantData(slices.Repeat([]byte("a"), min(dataLen, tt.bodyLen-n)))
				}
			}
			reqStream.wantClosed("request is complete")
		})
	}
}

func TestServer103EarlyHints(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		body := []byte("some body")
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			h := w.Header()

			h.Add("Content-Length", "123") // Must be ignored
			h.Add("Link", "</style.css>; rel=preload; as=style")
			h.Add("Link", "</script.js>; rel=preload; as=script")
			w.WriteHeader(http.StatusEarlyHints)

			h.Add("Link", "</foo.js>; rel=preload; as=script")
			w.WriteHeader(http.StatusEarlyHints)

			w.Write(body)                        // Implicitly sends status 200.
			w.WriteHeader(http.StatusEarlyHints) // Should be a no-op.
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantHeaders(http.Header{
			":status": {"103"},
			"Link": {
				"</style.css>; rel=preload; as=style",
				"</script.js>; rel=preload; as=script",
			},
		})
		reqStream.wantHeaders(http.Header{
			":status": {"103"},
			"Link": {
				"</style.css>; rel=preload; as=style",
				"</script.js>; rel=preload; as=script",
				"</foo.js>; rel=preload; as=script",
			},
		})
		reqStream.wantSomeHeaders(http.Header{
			":status":        {"200"},
			"Content-Length": {"123"},
		})
		reqStream.wantData(body)
		reqStream.wantClosed("request is complete")
	})
}

func TestServer304NotModified(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotModified)
			if _, err := w.Write([]byte("body should not be allowed")); !errors.Is(err, http.ErrBodyNotAllowed) {
				t.Errorf("got %v error when calling Write after WriteHeader(304), want %v error", err, http.ErrBodyNotAllowed)
			}
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{":status": {"304"}})
		reqStream.wantClosed("request is complete")
	})
}

func TestServerInvalidPathHeader(t *testing.T) {
	for _, test := range []struct {
		name string
		path string
	}{{
		name: "empty",
		path: "",
	}, {
		name: "invalid char",
		path: "\x00",
	}, {
		name: "absolute url",
		path: "https://example.com/",
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			ts := newTestServer(t, nil)
			tc := ts.connect()
			tc.greet()

			reqStream := tc.newStream(streamTypeRequest)
			reqStream.writeHeaders(requestHeader(http.Header{
				":path": []string{test.path},
			}))
			reqStream.wantError(quic.StreamError(errH3MessageError))
		})
	}
}

func TestServerOptionsMethod(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(http.Header{
			":method": []string{"OPTIONS"},
			":path":   []string{"*"},
		}))
		reqStream.wantSomeHeaders(http.Header{
			":status": {"200"},
		})
	})
}

func TestServerPastWriteDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctl := http.NewResponseController(w)
			io.WriteString(w, "one")
			if err := ctl.Flush(); err != nil {
				t.Errorf("Flush() = %v, want nil", err)
			}
			time.Sleep(time.Second) // T+1.
			// Set past write deadline. Write should fail.
			if err := ctl.SetWriteDeadline(time.Now().Add(-10 * time.Second)); err != nil {
				t.Errorf("SetWriteDeadline() = %v, want nil", err)
			}
			var err error
			_, err = io.WriteString(w, "x")
			if err == nil {
				err = ctl.Flush()
			}
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("got write err %v, want %v", err, os.ErrDeadlineExceeded)
			}

			// Extending the write deadline after it's exceeded should have no effect (sticky).
			if err := ctl.SetWriteDeadline(time.Now().Add(10 * time.Second)); err != nil {
				t.Errorf("SetWriteDeadline() = %v, want nil", err)
			}
			_, err = io.WriteString(w, "x")
			if err == nil {
				err = ctl.Flush()
			}
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("got write err %v (after extend), want %v", err, os.ErrDeadlineExceeded)
			}
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData([]byte("one"))
		time.Sleep(2 * time.Second) // T+2.
		synctest.Wait()
		reqStream.wantError(quic.StreamError(errH3RequestCancelled))
	})
}

func TestServerFutureWriteDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctl := http.NewResponseController(w)
			io.WriteString(w, "one")
			if err := ctl.Flush(); err != nil {
				t.Errorf("Flush() = %v, want nil", err)
			}

			// Set future deadline at T+1. Write should succeed.
			if err := ctl.SetWriteDeadline(time.Now().Add(time.Second)); err != nil {
				t.Errorf("SetWriteDeadline() = %v, want nil", err)
			}
			io.WriteString(w, "two")
			if err := ctl.Flush(); err != nil {
				t.Errorf("Flush() = %v, want nil", err)
			}

			// Extend deadline to T+3, before it expires.
			if err := ctl.SetWriteDeadline(time.Now().Add(3 * time.Second)); err != nil {
				t.Errorf("SetWriteDeadline() = %v, want nil", err)
			}
			// Sleep till T+2. Write should succeed since the deadline is T+3.
			time.Sleep(2 * time.Second)
			io.WriteString(w, "three")
			if err := ctl.Flush(); err != nil {
				t.Errorf("Flush() = %v, want nil", err)
			}

			// Sleep till T+4. Write should fail since deadline is T+3.
			time.Sleep(2 * time.Second)
			var err error
			_, err = io.WriteString(w, "x")
			if err == nil {
				err = ctl.Flush()
			}
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("got write err %v, want %v", err, os.ErrDeadlineExceeded)
			}

			// Extending the write deadline after it's exceeded should have no effect (sticky).
			if err := ctl.SetWriteDeadline(time.Time{}); err != nil {
				t.Errorf("SetWriteDeadline() = %v, want nil", err)
			}
			_, err = io.WriteString(w, "x")
			if err == nil {
				err = ctl.Flush()
			}
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("got write err %v (after extend), want %v", err, os.ErrDeadlineExceeded)
			}
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData([]byte("one"))
		reqStream.wantData([]byte("two"))
		time.Sleep(3 * time.Second) // T+3. After "three" is written.
		reqStream.wantData([]byte("three"))
		time.Sleep(3 * time.Second) // T+6. After server exceeds deadline.
		reqStream.wantError(quic.StreamError(errH3RequestCancelled))
	})
}

func TestServerPastReadDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctl := http.NewResponseController(w)
			b := make([]byte, 3)
			if _, err := io.ReadFull(r.Body, b); err != nil || string(b) != "one" {
				t.Errorf("Read() got (%q, %v), want (%q, nil)", b, err, "one")
			}
			// Set past read deadline. Read should fail.
			if err := ctl.SetReadDeadline(time.Now().Add(-10 * time.Second)); err != nil {
				t.Errorf("SetReadDeadline() = %v, want nil", err)
			}
			_, err := io.ReadAll(r.Body)
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("got read err %v, want %v", err, os.ErrDeadlineExceeded)
			}

			// Extending the read deadline after it's exceeded should have no effect (sticky).
			if err := ctl.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
				t.Errorf("SetReadDeadline() = %v, want nil", err)
			}
			_, err = io.ReadAll(r.Body)
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("got read err %v (after extend), want %v", err, os.ErrDeadlineExceeded)
			}
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.writeData([]byte("one"))
		synctest.Wait()
	})
}

func TestServerFutureReadDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ts := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctl := http.NewResponseController(w)
			b := make([]byte, 3)
			if _, err := io.ReadFull(r.Body, b); err != nil || string(b) != "one" {
				t.Errorf("Read() got (%q, %v), want (%q, nil)", b, err, "one")
			}

			// Set future deadline at T+2s. Read should succeed.
			if err := ctl.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
				t.Errorf("SetReadDeadline() = %v, want nil", err)
			}
			b2 := make([]byte, 3)
			if _, err := io.ReadFull(r.Body, b2); err != nil || string(b2) != "two" {
				t.Errorf("Read() got (%q, %v), want (%q, nil)", b2, err, "two")
			}

			// Extend deadline to T+5s, before it expires.
			if err := ctl.SetReadDeadline(time.Now().Add(4 * time.Second)); err != nil {
				t.Errorf("SetReadDeadline() = %v, want nil", err)
			}
			// Sleep till T+3. Read should succeed since the deadline is T+5.
			time.Sleep(2 * time.Second)
			b3 := make([]byte, 5)
			if _, err := io.ReadFull(r.Body, b3); err != nil || string(b3) != "three" {
				t.Errorf("Read() got (%q, %v), want (%q, nil)", b3, err, "three")
			}

			// Sleep till T+6. Read should fail since deadline has passed.
			time.Sleep(3 * time.Second)
			_, err := io.ReadAll(r.Body)
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("got read err %v, want %v", err, os.ErrDeadlineExceeded)
			}

			// Extending the read deadline after it's exceeded should have no effect (sticky).
			if err := ctl.SetReadDeadline(time.Time{}); err != nil {
				t.Errorf("SetReadDeadline() = %v, want nil", err)
			}
			_, err = io.ReadAll(r.Body)
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("got read err %v (after extend), want %v", err, os.ErrDeadlineExceeded)
			}
		}))
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.writeData([]byte("one"))

		time.Sleep(time.Second)
		reqStream.writeData([]byte("two")) // T+1.
		synctest.Wait()

		time.Sleep(time.Second)
		reqStream.writeData([]byte("three")) // T+2.
		synctest.Wait()

		time.Sleep(4 * time.Second) // Advance to T+6 for server handler to complete.
	})
}

func TestServerReadHeaderTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		timeout := 10 * time.Second
		ts := newTestServer(t, nil)
		ts.s.srv1.ReadHeaderTimeout = timeout
		tc := ts.connect()
		tc.greet()

		// Write some part of the header, but never finish sending it.
		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeVarint(int64(frameTypeHeaders))
		if err := reqStream.Flush(); err != nil {
			t.Fatalf("Flush() = %v, want nil", err)
		}

		// A stream error should be sent to the client as soon as the timeout
		// is reached. Server handler should not be called.
		time.Sleep(timeout - 1)
		reqStream.wantIdle("timeout has not been reached")
		time.Sleep(1)
		reqStream.wantError(quic.StreamError(errH3RequestRejected))
		if tc.nextHandlerCall() != nil {
			t.Error("server handler should not be called")
		}
	})
}

func TestServerReadTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		timeout := 10 * time.Second
		ts := newTestServer(t, nil)
		ts.s.srv1.ReadTimeout = timeout
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.writeData([]byte("some body"))
		call := tc.nextHandlerCall()

		// Read within the server handler should succeed prior to timeout.
		time.Sleep(timeout - 1)
		synctest.Wait()
		if _, err := call.req.Body.Read(make([]byte, 1)); err != nil {
			t.Errorf("Read() before timeout = %v, want nil", err)
		}

		// Read within the server handler should fail once timeout is reached.
		// Stream error should not be sent to the client, as it is up to the
		// server handler to decide how it wants to deal with its inability to
		// read the request body.
		time.Sleep(1)
		synctest.Wait()
		if _, err := call.req.Body.Read(make([]byte, 1)); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("Read() after timeout = %v, want os.ErrDeadlineExceeded", err)
		}
		call.w.Write([]byte("some body"))
		call.exit()
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData([]byte("some body"))
		reqStream.wantClosed("clean close expected")
	})
}

func TestServerReadTimeoutInProgress(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		timeout := 10 * time.Second
		ts := newTestServer(t, nil)
		ts.s.srv1.ReadTimeout = timeout
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.Flush()
		call := tc.nextHandlerCall()

		// Read will block due to the client having sent no body, thus
		// advancing synctest's time.
		start := time.Now()
		_, err := call.req.Body.Read(make([]byte, 1))
		if !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("Read error = %v, want os.ErrDeadlineExceeded", err)
		}
		if got, want := time.Since(start), timeout; got != want {
			t.Errorf("Read blocked for %v, want %v", got, want)
		}
		call.exit()
	})
}

func TestServerWriteTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		timeout := 10 * time.Second
		ts := newTestServer(t, nil)
		ts.s.srv1.WriteTimeout = timeout
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		call := tc.nextHandlerCall()
		body := make([]byte, defaultBodyBufferCap+1)

		// Write within the server handler should succeed prior to timeout.
		time.Sleep(timeout - 1)
		synctest.Wait()
		if _, err := call.w.Write(body); err != nil {
			t.Errorf("Write() before timeout = %v, want nil", err)
		}
		call.w.(http.Flusher).Flush()
		reqStream.wantSomeHeaders(http.Header{":status": {"200"}})
		reqStream.wantData(body)
		reqStream.wantIdle("timeout has not been reached")

		// Write within the server handler should fail once timeout is reached.
		// A stream error should also be sent to the client.
		time.Sleep(1)
		synctest.Wait()
		if _, err := call.w.Write(body); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("Write() after timeout = %v, want os.ErrDeadlineExceeded", err)
		}
		call.exit()
		reqStream.wantError(quic.StreamError(errH3RequestCancelled))
	})
}

func TestServerWriteTimeoutInProgress(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		timeout := 10 * time.Second
		ts := newTestServer(t, nil)
		ts.s.srv1.WriteTimeout = timeout
		tc := ts.connect()
		tc.greet()

		reqStream := tc.newStream(streamTypeRequest)
		reqStream.writeHeaders(requestHeader(nil))
		reqStream.Flush()
		call := tc.nextHandlerCall()

		// Keep writing body endlessly. Eventually, it will get blocked due to
		// flow control, and start advancing synctest's time.
		start := time.Now()
		var err error
		for err == nil {
			_, err = call.w.Write([]byte("endless body"))
		}
		if !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("Write error = %v, want os.ErrDeadlineExceeded", err)
		}
		if got, want := time.Since(start), timeout; got != want {
			t.Errorf("Write blocked for %v, want %v", got, want)
		}
		call.exit()
	})
}

type testServer struct {
	t           testing.TB
	s           *server
	tn          testNet
	testHandler *testServerHandler
	*testQUICEndpoint

	addr netip.AddrPort
}

type testQUICEndpoint struct {
	t testing.TB
	e *quic.Endpoint
}

type testServerConn struct {
	ts *testServer

	*testQUICConn
	control   *testQUICStream
	localAddr netip.AddrPort
}

type testServerHandler struct {
	ts      *testServer
	callsMu sync.Mutex
	calls   []*serverHandlerCall
}

// serverHandlerCall is a call to testServerHandler's ServeHTTP method.
type serverHandlerCall struct {
	w         http.ResponseWriter
	req       *http.Request
	closeOnce sync.Once
	ch        chan func()
}

func newTestServer(t testing.TB, handler http.Handler) *testServer {
	t.Helper()
	ts := &testServer{
		t: t,
	}
	if handler == nil {
		ts.testHandler = &testServerHandler{
			ts:    ts,
			calls: []*serverHandlerCall{},
		}
		handler = ts.testHandler
	}
	ts.s = &server{
		srv1: &http.Server{},
	}
	e := ts.tn.newQUICEndpoint(t, &quic.Config{
		TLSConfig: testTLSConfig,
	})
	ts.addr = e.LocalAddr()
	go ts.s.serve(t.Context(), e, handler)
	return ts
}

func (ts *testServer) connect() *testServerConn {
	ts.t.Helper()
	config := &quic.Config{TLSConfig: testTLSConfig}
	e := ts.tn.newQUICEndpoint(ts.t, nil)
	qconn, err := e.Dial(ts.t.Context(), "udp", ts.addr.String(), config)
	if err != nil {
		ts.t.Fatal(err)
	}
	tc := &testServerConn{
		ts:           ts,
		testQUICConn: newTestQUICConn(ts.t, qconn),
		localAddr:    e.LocalAddr(),
	}
	synctest.Wait()
	return tc
}

// greet performs initial connection handshaking with the server.
func (tc *testServerConn) greet() {
	// Client creates a control stream.
	tc.control = tc.newStream(streamTypeControl)
	tc.control.writeVarint(int64(frameTypeSettings))
	tc.control.writeVarint(0) // size
	tc.control.Flush()
	synctest.Wait()
}

// nextHandlerCall returns the next handler call that has been initiated by tc.
// If there is no handler call, nil is returned.
func (tc *testServerConn) nextHandlerCall() *serverHandlerCall {
	h := tc.ts.testHandler
	if h == nil {
		tc.t.Fatal("nextHandlerCall is called for a testServer with non-nil handler")
	}
	tc.t.Helper()
	synctest.Wait()
	h.callsMu.Lock()
	defer h.callsMu.Unlock()
	for i, call := range h.calls {
		if call.req.RemoteAddr == tc.localAddr.String() {
			h.calls = append(h.calls[:i], h.calls[i+1:]...)
			return call
		}
	}
	return nil
}

func (h *testServerHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	call := &serverHandlerCall{
		w:   w,
		req: req,
		ch:  make(chan func()),
	}
	h.ts.t.Cleanup(call.exit)
	h.callsMu.Lock()
	h.calls = append(h.calls, call)
	h.callsMu.Unlock()
	for f := range call.ch {
		f()
	}
}

// do executes f in the handler's goroutine.
func (call *serverHandlerCall) do(f func(http.ResponseWriter, *http.Request)) {
	donec := make(chan struct{})
	call.ch <- func() {
		defer close(donec)
		f(call.w, call.req)
	}
	<-donec
}

// exit causes the handler to return.
func (call *serverHandlerCall) exit() {
	call.closeOnce.Do(func() {
		close(call.ch)
	})
}
