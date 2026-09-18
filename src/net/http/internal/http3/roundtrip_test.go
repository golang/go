// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"net/http"
	"net/http/httptrace"
	"net/textproto"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"testing/synctest"

	"golang.org/x/net/quic"
)

func TestRoundTripSimple(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		req, _ := http.NewRequest("GET", "https://example.tld/", nil)
		req.Header["User-Agent"] = nil
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)
		st.wantSomeHeaders(http.Header{
			":authority": []string{"example.tld"},
			":method":    []string{"GET"},
			":path":      []string{"/"},
			":scheme":    []string{"https"},
		})
		st.writeHeaders(http.Header{
			":status":       []string{"200"},
			"x-some-header": []string{"value"},
		})
		rt.wantStatus(200)
		rt.wantHeaders(http.Header{
			"X-Some-Header": []string{"value"},
		})
	})
}

func TestRoundTripWithBadHeaders(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		req, _ := http.NewRequest("GET", "https://example.tld/", nil)
		req.Header["Invalid\nHeader"] = []string{"x"}
		rt := tc.roundTrip(req)
		rt.wantError("RoundTrip fails when request contains invalid headers")
	})
}

func TestRoundTripWithUnknownFrame(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		req, _ := http.NewRequest("GET", "https://example.tld/", nil)
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)

		// Write an unknown frame type before the response HEADERS.
		data := "frame content"
		st.writeVarint(0x1f + 0x21)      // reserved frame type
		st.writeVarint(int64(len(data))) // size
		st.Write([]byte(data))

		st.writeHeaders(http.Header{
			":status": []string{"200"},
		})
		rt.wantStatus(200)
	})
}

func TestRoundTripWithInvalidPushPromise(t *testing.T) {
	// "A client MUST treat receipt of a PUSH_PROMISE frame that contains
	// a larger push ID than the client has advertised as a connection error of H3_ID_ERROR."
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-7.2.5-5
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		req, _ := http.NewRequest("GET", "https://example.tld/", nil)
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)

		// Write a PUSH_PROMISE frame.
		// Since the client hasn't indicated willingness to accept pushes,
		// this is a connection error.
		st.writePushPromise(0, http.Header{
			":path": []string{"/foo"},
		})
		rt.wantError("RoundTrip fails after receiving invalid PUSH_PROMISE")
		tc.wantClosed(
			"push ID exceeds client's MAX_PUSH_ID",
			errH3IDError,
		)
	})
}

func TestRoundTripResponseContentLength(t *testing.T) {
	for _, test := range []struct {
		name              string
		respHeader        http.Header
		wantContentLength int64
		wantError         bool
	}{{
		name: "valid",
		respHeader: http.Header{
			":status":        []string{"200"},
			"content-length": []string{"100"},
		},
		wantContentLength: 100,
	}, {
		name: "absent",
		respHeader: http.Header{
			":status": []string{"200"},
		},
		wantContentLength: -1,
	}, {
		name: "unparsable",
		respHeader: http.Header{
			":status":        []string{"200"},
			"content-length": []string{"1 1"},
		},
		wantError: true,
	}, {
		name: "duplicated",
		respHeader: http.Header{
			":status":        []string{"200"},
			"content-length": []string{"500", "500", "500"},
		},
		wantContentLength: 500,
	}, {
		name: "inconsistent",
		respHeader: http.Header{
			":status":        []string{"200"},
			"content-length": []string{"1", "2"},
		},
		wantError: true,
	}, {
		// 204 responses aren't allowed to contain a Content-Length header.
		// We just ignore it.
		name: "204",
		respHeader: http.Header{
			":status":        []string{"204"},
			"content-length": []string{"100"},
		},
		wantContentLength: -1,
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			tc := newTestClientConn(t)
			tc.greet()

			req, _ := http.NewRequest("GET", "https://example.tld/", nil)
			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)
			st.wantHeaders(nil)
			st.writeHeaders(test.respHeader)
			if test.wantError {
				rt.wantError("invalid content-length in response")
				return
			}
			if got, want := rt.response().ContentLength, test.wantContentLength; got != want {
				t.Errorf("Response.ContentLength = %v, want %v", got, want)
			}
		})
	}
}

func TestRoundTripMalformedResponses(t *testing.T) {
	for _, test := range []struct {
		name       string
		respHeader http.Header
	}{{
		name: "duplicate :status",
		respHeader: http.Header{
			":status": {"200", "204"},
		},
	}, {
		name: "unparsable :status",
		respHeader: http.Header{
			":status": {"frogpants"},
		},
	}, {
		name: "undefined pseudo-header",
		respHeader: http.Header{
			":status":  {"200"},
			":unknown": {"x"},
		},
	}, {
		name:       "no :status",
		respHeader: http.Header{},
	}, {
		name: "header name with control character",
		respHeader: http.Header{
			":status":             {"200"},
			"name\nevilinjection": {"Value"},
		},
	}, {
		name: "header name with uppercase character",
		respHeader: http.Header{
			":status": {"200"},
			"nAme":    {"Value"},
		},
	}, {
		name:       "pseudo-header name with control character",
		respHeader: http.Header{":status\nevilinjection": {"200"}},
	}, {
		name:       "pseudo-header name with uppercase character",
		respHeader: http.Header{":stAtus": {"200"}},
	}, {
		name: "header value with control character",
		respHeader: http.Header{
			":status": {"200"},
			"name":    {"Value\nEvilInjection"},
		},
	}, {
		name:       "pseudo-header value with control character",
		respHeader: http.Header{":status": {"200\nEvilInjection"}},
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			tc := newTestClientConn(t)
			tc.greet()

			req, _ := http.NewRequest("GET", "https://example.tld/", nil)
			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)
			st.wantHeaders(nil)
			st.writeHeadersRaw(test.respHeader)
			rt.wantError("malformed response")
		})
	}
}

func TestRoundTripCrumbledCookiesInResponse(t *testing.T) {
	// "If a decompressed field section contains multiple cookie field lines,
	// these MUST be concatenated into a single byte string [...]"
	// using the two-byte delimiter of "; "''
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-4.2.1-2
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		req, _ := http.NewRequest("GET", "https://example.tld/", nil)
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)
		st.writeHeaders(http.Header{
			":status": []string{"200"},
			"cookie":  []string{"a=1", "b=2; c=3", "d=4"},
		})
		rt.wantStatus(200)
		rt.wantHeaders(http.Header{
			"Cookie": []string{"a=1; b=2; c=3; d=4"},
		})
	})
}

func TestRoundTripRequestBodySent(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		bodyr, bodyw := io.Pipe()

		req, _ := http.NewRequest("GET", "https://example.tld/", bodyr)
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)

		bodyw.Write([]byte{0, 1, 2, 3, 4})
		st.wantData([]byte{0, 1, 2, 3, 4})

		bodyw.Write([]byte{5, 6, 7})
		st.wantData([]byte{5, 6, 7})

		bodyw.Close()
		st.wantClosed("request body sent")

		st.writeHeaders(http.Header{
			":status": []string{"200"},
		})
		rt.wantStatus(200)
		rt.response().Body.Close()
	})
}

func TestRoundTripRequestBodyErrors(t *testing.T) {
	for _, test := range []struct {
		name          string
		body          io.Reader
		contentLength int64
	}{{
		name:          "too short",
		contentLength: 10,
		body:          bytes.NewReader([]byte{0, 1, 2, 3, 4}),
	}, {
		name:          "too long",
		contentLength: 5,
		body:          bytes.NewReader([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
	}, {
		name: "read error",
		body: io.MultiReader(
			bytes.NewReader([]byte{0, 1, 2, 3, 4}),
			&testReader{
				readFunc: func([]byte) (int, error) {
					return 0, errors.New("read error")
				},
			},
		),
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			tc := newTestClientConn(t)
			tc.greet()

			req, _ := http.NewRequest("GET", "https://example.tld/", test.body)
			req.ContentLength = test.contentLength
			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)

			// The Transport should send some number of frames before detecting an
			// error in the request body and aborting the request.
			synctest.Wait()
			for {
				_, err := st.readFrameHeader()
				if err != nil {
					var code quic.StreamError
					if !errors.As(err, &code) {
						t.Fatalf("request stream closed with error %v: want QUIC stream error", err)
					}
					break
				}
				if err := st.discardFrame(); err != nil {
					t.Fatalf("discardFrame: %v", err)
				}
			}

			// RoundTrip returns with an error.
			rt.wantError("request fails due to body error")
		})
	}
}

func TestRoundTripRequestBodyErrorAfterHeaders(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		bodyr, bodyw := io.Pipe()
		req, _ := http.NewRequest("GET", "https://example.tld/", bodyr)
		req.ContentLength = 10
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)

		// Server sends response headers, and RoundTrip returns.
		// The request body hasn't been sent yet.
		st.wantHeaders(nil)
		st.writeHeaders(http.Header{
			":status": []string{"200"},
		})
		rt.wantStatus(200)

		// Write too many bytes to the request body, triggering a request error.
		bodyw.Write(make([]byte, req.ContentLength+1))

		//io.Copy(io.Discard, st)
		st.wantError(quic.StreamError(errH3InternalError))

		if err := rt.response().Body.Close(); err == nil {
			t.Fatalf("Response.Body.Close() = %v, want error", err)
		}
	})
}

func TestRoundTripRequestBodyIgnored(t *testing.T) {
	for _, tt := range []struct {
		name            string
		sendPartialBody bool
	}{{
		name:            "after partial body",
		sendPartialBody: true,
	}, {
		name:            "before any body",
		sendPartialBody: false,
	}} {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			tc := newTestClientConn(t)
			tc.greet()

			bodyr, bodyw := io.Pipe()
			req, _ := http.NewRequest("POST", "https://example.tld/", bodyr)
			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)
			st.wantHeaders(nil)

			if tt.sendPartialBody {
				bodyw.Write([]byte("hello"))
				st.wantData([]byte("hello"))
			}

			// Server stops reading the request because it has enough
			// information already to construct its response.
			st.CloseRead(uint64(errH3NoError))
			st.writeHeaders(http.Header{
				":status": {"200"},
			})
			synctest.Wait()

			// Further writes will fail due to the stream being reset after the
			// server closes its read. In this case, the transport should
			// gracefully stop writing and surface the response it has
			// received, rather than erroring out.
			bodyw.Write([]byte("hello again"))
			synctest.Wait()
			rt.wantStatus(200)
			if err := rt.response().Body.Close(); err != nil {
				t.Fatalf("Response.Body.Close() = %v, want nil", err)
			}
		})
	}
}

// TestRoundTripClosesRequestBodyOnError verifies that a RoundTrip which fails
// closes the request body before returning, rather than leaving the body
// writer goroutine to close it at some later point.
//
// net/http inspects the request body as soon as RoundTrip returns to decide
// whether it needs to close the body itself, so a close which happens
// concurrently with the return is too late. See golang/go#60041.
func TestRoundTripClosesRequestBodyOnError(t *testing.T) {
	for _, tt := range []struct {
		name          string
		sendExpect100 bool
	}{{
		// The body writer has started, and is blocked reading from the body.
		name:          "body writer started",
		sendExpect100: false,
	}, {
		// The body writer never started, because the client is still waiting
		// for the server to send 100 Continue.
		name:          "body writer not started",
		sendExpect100: true,
	}} {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			tc := newTestClientConn(t)
			tc.greet()

			body := newTestRequestBody()
			req, _ := http.NewRequest("POST", "https://example.tld/", body)
			if tt.sendExpect100 {
				req.Header.Set("Expect", "100-continue")
			}
			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)
			st.wantHeaders(nil)

			// The server resets the request stream, failing the request.
			st.Reset(uint64(errH3InternalError))
			rt.wantError("server reset the request stream")

			if got := body.closeCount(); got != 1 {
				t.Errorf("Request.Body closed %v times when RoundTrip returned, want 1", got)
			}
		})
	}
}

func TestRoundTripExpect100Continue(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var callCount1xx, callCount100, callCount100Wait int
		trace := &httptrace.ClientTrace{
			Got1xxResponse: func(code int, header textproto.MIMEHeader) error {
				callCount1xx++
				return nil
			},
			Got100Continue: func() {
				callCount100++
			},
			Wait100Continue: func() {
				callCount100Wait++
			},
		}

		tc := newTestClientConn(t)
		tc.greet()
		clientBody := []byte("client's body that will be sent later")
		serverBody := []byte("server's body")

		// Client sends an Expect: 100-continue request.
		req, _ := http.NewRequestWithContext(httptrace.WithClientTrace(t.Context(), trace), "GET", "https://example.tld/", bytes.NewBuffer(clientBody))
		req.Header = http.Header{"Expect": {"100-continue"}}
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)

		// Server reads the header.
		st.wantHeaders(nil)
		st.wantIdle("client has yet to send its body")

		// Server responds with HTTP status 100.
		st.writeHeaders(http.Header{
			":status": []string{"100"},
		})

		// Client sends its body after receiving HTTP status 100 response.
		st.wantData(clientBody)

		// The server sends its response after getting the client's body.
		st.writeHeaders(http.Header{
			":status": []string{"200"},
		})
		st.writeData(serverBody)
		st.CloseWrite()

		// Client receives the response from server.
		rt.wantStatus(200)
		rt.wantBody(serverBody)

		gotCount := []int{callCount1xx, callCount100, callCount100Wait}
		if !slices.Equal(gotCount, []int{1, 1, 1}) {
			t.Errorf("Got1xxResponse, Got100Continue, and Wait100Continue was called %v times respectively, want [1 1 1]", gotCount)
		}
	})
}

// TestRoundTripInformationalHeaders verifies that informational 1xx statuses
// are never treated as the final status of a response.
func TestRoundTripInformationalHeaders(t *testing.T) {
	for _, tt := range []struct {
		name          string
		sendExpect100 bool
		infoStatuses  []int
	}{
		{
			name:          "unexpected 100 without expect header",
			sendExpect100: false,
			infoStatuses:  []int{100},
		},
		{
			name:          "duplicate 100 continue",
			sendExpect100: true,
			infoStatuses:  []int{100, 100},
		},
		{
			name:          "interleaved 1xx and 100 continue",
			sendExpect100: true,
			infoStatuses:  []int{103, 100, 102},
		},
		{
			name:          "1xx with no 100 continue",
			sendExpect100: true, // Client sends Expect: 100-continue, but server never sends 100.
			infoStatuses:  []int{103, 102},
		},
	} {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			tc := newTestClientConn(t)
			tc.greet()

			body := []byte("request payload")
			req, _ := http.NewRequest("POST", "https://example.tld/", bytes.NewReader(body))
			if tt.sendExpect100 {
				req.Header.Set("Expect", "100-continue")
			}

			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)
			st.wantHeaders(nil)

			bodySent := !tt.sendExpect100
			if bodySent {
				st.wantData(body)
				st.wantClosed("body sent")
			}

			for _, status := range tt.infoStatuses {
				st.writeHeaders(http.Header{
					":status": {strconv.Itoa(status)},
				})
				if status == 100 && !bodySent {
					bodySent = true
					st.wantData(body)
					st.wantClosed("body sent after 100 continue")
				}
			}

			st.writeHeaders(http.Header{
				":status": {"200"},
			})
			st.writeData([]byte("response payload"))
			st.CloseWrite()

			rt.wantStatus(200)
			rt.wantBody([]byte("response payload"))
		})
	}
}

func TestRoundTripExpect100ContinueRejected(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var callCount1xx, callCount100, callCount100Wait int
		trace := &httptrace.ClientTrace{
			Got1xxResponse: func(code int, header textproto.MIMEHeader) error {
				callCount1xx++
				return nil
			},
			Got100Continue: func() {
				callCount100++
			},
			Wait100Continue: func() {
				callCount100Wait++
			},
		}

		tc := newTestClientConn(t)
		tc.greet()

		// Client sends an Expect: 100-continue request.
		req, _ := http.NewRequestWithContext(httptrace.WithClientTrace(t.Context(), trace), "GET", "https://example.tld/", bytes.NewBufferString("client's body"))
		req.Header = http.Header{"Expect": {"100-continue"}}
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)

		// Server reads the header.
		st.wantHeaders(nil)
		st.wantIdle("client has yet to send its body")

		// Server rejects it.
		st.writeHeaders(http.Header{
			":status": []string{"200"},
		})
		st.wantIdle("client does not send its body without getting status 100")
		serverBody := []byte("server's body")
		st.writeData(serverBody)
		st.CloseWrite()

		rt.wantStatus(200)
		rt.wantBody(serverBody)

		gotCount := []int{callCount1xx, callCount100, callCount100Wait}
		if !slices.Equal(gotCount, []int{0, 0, 1}) {
			t.Errorf("Got1xxResponse, Got100Continue, and Wait100Continue was called %v times respectively, want [0 0 1]", gotCount)
		}
	})
}

func TestRoundTripNoBodyClosesStream(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		req, _ := http.NewRequest("PUT", "https://example.tld/", nil)
		tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)

		st.wantHeaders(nil)
		st.wantClosed("no DATA frames to send")
	})
}

func TestRoundTripReadRespWithNoBody(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		// Case 1: we know response body is empty because the server closes the
		// write direction of the stream.
		req, _ := http.NewRequest("GET", "https://example.tld/", nil)
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)
		st.writeHeaders(http.Header{
			":status": {"200"},
		})
		st.CloseWrite()
		rt.wantStatus(200)
		st.wantClosed("request is complete")

		// Case 2: we know response body is empty because the server indicates
		// a Content-Length of 0.
		req, _ = http.NewRequest("GET", "https://example.tld/", nil)
		rt = tc.roundTrip(req)
		st = tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)
		st.writeHeaders(http.Header{
			":status":        {"200"},
			"content-length": {"0"},
		})
		rt.wantStatus(200)
		st.wantClosed("request is complete")

		// Case 3: we know response body is empty because we sent a HEAD
		// request.
		req, _ = http.NewRequest("HEAD", "https://example.tld/", nil)
		rt = tc.roundTrip(req)
		st = tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)
		st.writeHeaders(http.Header{
			":status":        {"200"},
			"content-length": {"1000"},
		})
		rt.wantStatus(200)
		st.wantClosed("request is complete")
	})
}

func TestRoundTripWriteTrailer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		var req *http.Request
		req, _ = http.NewRequest("POST", "https://example.tld/", io.MultiReader(
			testReader{readFunc: func(_ []byte) (int, error) {
				req.Trailer["Client-Trailer-A"] = []string{"valuea"}
				// Transport should not send undeclared trailer.
				req.Trailer["Undeclared-Trailer"] = []string{"undeclared"}
				return 0, io.EOF
			}},
			strings.NewReader("a body"),
			testReader{readFunc: func(_ []byte) (int, error) {
				req.Trailer["Client-Trailer-B"] = []string{"valueb"}
				// Transport should not send undeclared trailer.
				req.Trailer["Undeclared-Trailer"] = []string{"undeclared"}
				return 0, io.EOF
			}},
		))
		req.Trailer = http.Header{
			"Client-Trailer-A": nil,
			"Client-Trailer-B": nil,
		}
		tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)
		st.wantData([]byte("a body"))
		st.wantHeaders(http.Header{
			"Client-Trailer-A": {"valuea"},
			"Client-Trailer-B": {"valueb"},
		})
		st.wantClosed("request is complete")
	})
}

func TestRoundTripWriteTrailerNoBody(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		var req *http.Request
		req, _ = http.NewRequest("POST", "https://example.tld/", io.MultiReader(
			testReader{readFunc: func(_ []byte) (int, error) {
				req.Trailer["Client-Trailer-A"] = []string{"valuea"}
				// Transport should not send undeclared trailer.
				req.Trailer["Undeclared-Trailer"] = []string{"undeclared"}
				return 0, io.EOF
			}},
			testReader{readFunc: func(_ []byte) (int, error) {
				req.Trailer["Client-Trailer-B"] = []string{"valueb"}
				// Transport should not send undeclared trailer.
				req.Trailer["Undeclared-Trailer"] = []string{"undeclared"}
				return 0, io.EOF
			}},
		))
		req.Trailer = http.Header{
			"Client-Trailer-A": nil,
			"Client-Trailer-B": nil,
		}
		tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)
		st.wantHeaders(nil)
		st.wantHeaders(http.Header{
			"Client-Trailer-A": {"valuea"},
			"Client-Trailer-B": {"valueb"},
		})
		st.wantClosed("request is complete")
	})
}

func TestRoundTripReadTrailer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		var req *http.Request
		req, _ = http.NewRequest("GET", "https://example.tld/", nil)
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)

		st.wantHeaders(nil)
		st.writeHeaders(http.Header{
			":status": {"200"},
			"trailer": {"Server-Trailer-A, Server-Trailer-B", "server-trailer-c"}, // Should be canonicalized.
		})
		body := []byte("body from server")
		st.writeData(body)
		st.writeHeaders(http.Header{
			"server-trailer-a": {"valuea"},
			// Note that Server-Trailer-B is skipped.
			"server-trailer-c":   {"valuec"},
			"undeclared-trailer": {"undeclared"},
		})

		rt.wantStatus(200)
		// Trailer is stripped off from http.Response.Header and given in http.Response.Trailer.
		rt.wantHeaders(http.Header{})
		rt.wantTrailers(http.Header{
			"Server-Trailer-A": nil,
			"Server-Trailer-B": nil,
			"Server-Trailer-C": nil,
		})

		// Trailer updated after reading the body to EOF.
		rt.wantBody(body)
		rt.wantTrailers(http.Header{
			"Server-Trailer-A": {"valuea"},
			"Server-Trailer-B": nil,
			"Server-Trailer-C": {"valuec"},
			// Transport should accept undeclared trailers.
			"Undeclared-Trailer": {"undeclared"},
		})
		st.wantClosed("request is complete")
	})
}

func TestRoundTripReadTrailerNoBody(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		var req *http.Request
		req, _ = http.NewRequest("GET", "https://example.tld/", nil)
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)

		st.wantHeaders(nil)
		st.writeHeaders(http.Header{
			":status":        {"200"},
			"content-length": {"0"},
			"trailer":        {"Server-Trailer-A, Server-Trailer-B", "server-trailer-c"}, // Should be canonicalized.
		})
		st.writeHeaders(http.Header{
			"server-trailer-a": {"valuea"},
			// Note that Server-Trailer-B is skipped.
			"server-trailer-c":   {"valuec"},
			"undeclared-trailer": {"undeclared"},
		})

		rt.wantStatus(200)
		// Trailer is stripped off from http.Response.Header and given in http.Response.Trailer.
		rt.wantHeaders(http.Header{"Content-Length": {"0"}})
		rt.wantTrailers(http.Header{
			"Server-Trailer-A": nil,
			"Server-Trailer-B": nil,
			"Server-Trailer-C": nil,
		})

		// Trailer updated after reading the empty body to EOF.
		rt.wantBody(make([]byte, 0))
		rt.wantTrailers(http.Header{
			"Server-Trailer-A": {"valuea"},
			"Server-Trailer-B": nil,
			"Server-Trailer-C": {"valuec"},
			// Transport should accept undeclared trailers.
			"Undeclared-Trailer": {"undeclared"},
		})
		st.wantClosed("request is complete")
	})
}

func TestRoundTrip103EarlyHints(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		firstHeader := http.Header{
			":status": {"103"},
			"Link":    {"</style.css>; rel=preload; as=style"},
		}
		secondHeader := http.Header{
			":status": {"103"},
			"Link":    {"</style.css>; rel=preload; as=style", "</script.js>; rel=preload; as=script"},
		}

		var respCounter int
		trace := &httptrace.ClientTrace{
			Got1xxResponse: func(code int, header textproto.MIMEHeader) error {
				var wantHeader textproto.MIMEHeader
				switch respCounter {
				case 0:
					wantHeader = textproto.MIMEHeader(firstHeader)
				case 1:
					wantHeader = textproto.MIMEHeader(secondHeader)
				default:
					t.Error("Unexpected 1xx response")
				}
				wantHeader.Del(":status")
				if !reflect.DeepEqual(header, wantHeader) {
					t.Errorf("got %v early hints header, want %v", header, wantHeader)
				}
				respCounter++
				return nil
			},
		}
		req, _ := http.NewRequestWithContext(httptrace.WithClientTrace(t.Context(), trace), "GET", "https://example.tld/", nil)

		tc := newTestClientConn(t)
		tc.greet()
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)

		st.wantHeaders(nil)
		st.writeHeaders(firstHeader)
		st.writeHeaders(secondHeader)

		st.writeHeaders(http.Header{
			":status": {"200"},
		})
		body := []byte("some body")
		st.writeData(body)
		st.CloseWrite()

		rt.wantStatus(200)
		rt.wantBody(body)
		st.wantClosed("request is complete")
	})
}

func TestRoundTripGzipEnabled(t *testing.T) {
	tests := []struct {
		name     string
		explicit bool
	}{
		{
			name:     "transparent",
			explicit: false,
		},
		{
			name:     "explicit",
			explicit: true,
		},
	}
	for _, tt := range tests {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			tc := newTestClientConn(t)
			tc.greet()

			req, _ := http.NewRequest("GET", "https://example.tld/", nil)
			if tt.explicit {
				req.Header.Set("Accept-Encoding", "gzip")
			}
			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)

			// Verify that client sends Accept-Encoding: gzip.
			st.wantSomeHeaders(http.Header{
				"Accept-Encoding": []string{"gzip"},
			})

			// Server responds with gzip.
			var buf bytes.Buffer
			gw := gzip.NewWriter(&buf)
			gw.Write([]byte("hello world"))
			gw.Close()
			st.writeHeaders(http.Header{
				":status":          []string{"200"},
				"content-encoding": []string{"gzip"},
				"content-length":   []string{strconv.Itoa(buf.Len())},
			})
			st.writeData(buf.Bytes())
			st.CloseWrite()

			rt.wantStatus(200)

			if tt.explicit {
				// When user explicitly sets gzip, the server response should
				// be given as is.
				rt.wantBody(buf.Bytes())
				resp, err := rt.result()
				if err != nil {
					t.Fatal(err)
				}
				if resp.Header.Get("Content-Encoding") != "gzip" {
					t.Errorf("Content-Encoding = %q, want gzip", resp.Header.Get("Content-Encoding"))
				}
				if resp.Header.Get("Content-Length") != strconv.Itoa(buf.Len()) {
					t.Errorf("Content-Length = %q, want %d", resp.Header.Get("Content-Length"), buf.Len())
				}
				if resp.ContentLength != int64(buf.Len()) {
					t.Errorf("ContentLength = %d, want %d", resp.ContentLength, buf.Len())
				}
				if resp.Uncompressed {
					t.Errorf("Uncompressed = true, want false")
				}
			} else {
				// When gzip is transparently set, we automatically decode the
				// response body, and make sure stale information about the
				// gzip content length and encoding are updated.
				rt.wantBody([]byte("hello world"))
				resp, err := rt.result()
				if err != nil {
					t.Fatal(err)
				}
				if resp.Header.Get("Content-Encoding") != "" {
					t.Errorf("Content-Encoding = %q, want empty", resp.Header.Get("Content-Encoding"))
				}
				if resp.Header.Get("Content-Length") != "" {
					t.Errorf("Content-Length = %q, want empty", resp.Header.Get("Content-Length"))
				}
				if resp.ContentLength != -1 {
					t.Errorf("ContentLength = %d, want -1", resp.ContentLength)
				}
				if !resp.Uncompressed {
					t.Errorf("Uncompressed = false, want true")
				}
			}
		})
	}
}

func TestRoundTripGzipDisabled(t *testing.T) {
	tests := []struct {
		name  string
		setup func(tc *testClientConn, req *http.Request, wantHeaders http.Header)
	}{
		{
			name: "explicitly disabled",
			setup: func(tc *testClientConn, req *http.Request, wantHeaders http.Header) {
				tc.tr.tr1.DisableCompression = true
			},
		},
		{
			name: "HEAD request",
			setup: func(tc *testClientConn, req *http.Request, wantHeaders http.Header) {
				req.Method = "HEAD"
				wantHeaders.Set(":method", "HEAD")
			},
		},
		{
			name: "contains Range header",
			setup: func(tc *testClientConn, req *http.Request, wantHeaders http.Header) {
				req.Header.Set("Range", "bytes=0-10")
				wantHeaders.Set("Range", "bytes=0-10")
			},
		},
		{
			name: "contains Accept-Encoding-identity header",
			setup: func(tc *testClientConn, req *http.Request, wantHeaders http.Header) {
				req.Header.Set("Accept-Encoding", "identity")
				wantHeaders.Set("Accept-Encoding", "identity")
			},
		},
	}
	for _, tt := range tests {
		synctestSubtest(t, tt.name, func(t *testing.T) {
			tc := newTestClientConn(t)
			req, _ := http.NewRequest("GET", "https://example.tld/", nil)
			wantHeaders := http.Header{
				":authority": []string{"example.tld"},
				":method":    []string{"GET"},
				":path":      []string{"/"},
				":scheme":    []string{"https"},
				"User-Agent": []string{"Go-http-client/3.0"},
			}
			tt.setup(tc, req, wantHeaders)
			tc.greet()

			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)

			// Verify that client does not send Accept-Encoding: gzip.
			st.wantHeaders(wantHeaders)

			st.writeHeaders(http.Header{
				":status": []string{"200"},
			})
			rt.wantStatus(200)
		})
	}
}

func TestRoundTripGzipWithTrailers(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tc := newTestClientConn(t)
		tc.greet()

		req, _ := http.NewRequest("GET", "https://example.tld/", nil)
		rt := tc.roundTrip(req)
		st := tc.wantStream(streamTypeRequest)

		// Verify that client sends Accept-Encoding: gzip.
		st.wantSomeHeaders(http.Header{
			"Accept-Encoding": []string{"gzip"},
		})

		// Server responds with gzip and trailer declaration.
		var buf bytes.Buffer
		gw := gzip.NewWriter(&buf)
		gw.Write([]byte("hello world"))
		gw.Close()
		st.writeHeaders(http.Header{
			":status":          []string{"200"},
			"content-encoding": []string{"gzip"},
			"trailer":          []string{"Server-Trailer-A"},
		})
		st.writeData(buf.Bytes())
		st.writeHeaders(http.Header{
			"server-trailer-a": {"valuea"},
		})
		st.CloseWrite()

		rt.wantStatus(200)
		rt.wantTrailers(http.Header{
			"Server-Trailer-A": nil,
		})
		rt.wantBody([]byte("hello world"))
		rt.wantTrailers(http.Header{
			"Server-Trailer-A": {"valuea"},
		})
		st.wantClosed("request is complete")
	})
}
