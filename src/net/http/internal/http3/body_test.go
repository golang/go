// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"testing"
)

// TestReadData tests servers reading request bodies, and clients reading response bodies.
func TestReadData(t *testing.T) {
	// These tests consist of a series of steps,
	// where each step is either something arriving on the stream
	// or the client/server reading from the body.
	type (
		// HEADERS frame arrives (headers).
		receiveHeaders struct {
			contentLength int64 // -1 for no content-length
		}
		// DATA frame header arrives.
		receiveDataHeader struct {
			size int64
		}
		// DATA frame content arrives.
		receiveData struct {
			size int64
		}
		// HEADERS frame arrives (trailers).
		receiveTrailers struct{}
		// Some other frame arrives.
		receiveFrame struct {
			ftype frameType
			data  []byte
		}
		// Stream closed, ending the body.
		receiveEOF struct{}
		// Server reads from Request.Body, or client reads from Response.Body.
		wantBody struct {
			size int64
			eof  bool
		}
		// Check that reading the body results in non-EOF error.
		wantError struct {
			// If err is not nil, also check that the error received is err.
			err error
		}
		// Close the body.
		closeBody struct{}
	)
	for _, test := range []struct {
		name       string
		respHeader http.Header
		steps      []any
		wantError  bool
	}{{
		name: "no content length",
		steps: []any{
			receiveHeaders{contentLength: -1},
			receiveDataHeader{size: 10},
			receiveData{size: 10},
			receiveEOF{},
			wantBody{size: 10, eof: true},
		},
	}, {
		name: "valid content length",
		steps: []any{
			receiveHeaders{contentLength: 10},
			receiveDataHeader{size: 10},
			receiveData{size: 10},
			receiveEOF{},
			wantBody{size: 10, eof: true},
		},
	}, {
		name: "data frame exceeds content length",
		steps: []any{
			receiveHeaders{contentLength: 5},
			receiveDataHeader{size: 10},
			receiveData{size: 10},
			wantError{},
		},
	}, {
		name: "data frame after all content read",
		steps: []any{
			receiveHeaders{contentLength: 5},
			receiveDataHeader{size: 5},
			receiveData{size: 5},
			wantBody{size: 5},
			receiveDataHeader{size: 1},
			receiveData{size: 1},
			wantError{},
		},
	}, {
		name: "content length too long",
		steps: []any{
			receiveHeaders{contentLength: 10},
			receiveDataHeader{size: 5},
			receiveData{size: 5},
			receiveEOF{},
			wantBody{size: 5},
			wantError{},
		},
	}, {
		name: "stream ended by trailers",
		steps: []any{
			receiveHeaders{contentLength: -1},
			receiveDataHeader{size: 5},
			receiveData{size: 5},
			receiveTrailers{},
			wantBody{size: 5, eof: true},
		},
	}, {
		name: "trailers and content length too long",
		steps: []any{
			receiveHeaders{contentLength: 10},
			receiveDataHeader{size: 5},
			receiveData{size: 5},
			wantBody{size: 5},
			receiveTrailers{},
			wantError{},
		},
	}, {
		name: "unknown frame before headers",
		steps: []any{
			receiveFrame{
				ftype: 0x1f + 0x21, // reserved frame type
				data:  []byte{1, 2, 3, 4},
			},
			receiveHeaders{contentLength: -1},
			receiveDataHeader{size: 10},
			receiveData{size: 10},
			wantBody{size: 10},
		},
	}, {
		name: "unknown frame after headers",
		steps: []any{
			receiveHeaders{contentLength: -1},
			receiveFrame{
				ftype: 0x1f + 0x21, // reserved frame type
				data:  []byte{1, 2, 3, 4},
			},
			receiveDataHeader{size: 10},
			receiveData{size: 10},
			wantBody{size: 10},
		},
	}, {
		name: "invalid frame",
		steps: []any{
			receiveHeaders{contentLength: -1},
			receiveFrame{
				ftype: frameTypeSettings, // not a valid frame on this stream
				data:  []byte{1, 2, 3, 4},
			},
			wantError{},
		},
	}, {
		name: "data frame consumed by several reads",
		steps: []any{
			receiveHeaders{contentLength: -1},
			receiveDataHeader{size: 16},
			receiveData{size: 16},
			wantBody{size: 2},
			wantBody{size: 4},
			wantBody{size: 8},
			wantBody{size: 2},
		},
	}, {
		name: "read multiple frames",
		steps: []any{
			receiveHeaders{contentLength: -1},
			receiveDataHeader{size: 2},
			receiveData{size: 2},
			receiveDataHeader{size: 4},
			receiveData{size: 4},
			receiveDataHeader{size: 8},
			receiveData{size: 8},
			wantBody{size: 2},
			wantBody{size: 4},
			wantBody{size: 8},
		},
	}, {
		name: "read after body close",
		steps: []any{
			receiveHeaders{contentLength: -1},
			receiveDataHeader{size: 2},
			receiveData{size: 2},
			receiveDataHeader{size: 4},
			receiveData{size: 4},
			receiveDataHeader{size: 8},
			receiveData{size: 8},
			wantBody{size: 2},
			wantBody{size: 4},
			closeBody{},
			wantError{err: net.ErrClosed},
		},
	}} {

		runTest := func(t testing.TB, h http.Header, st *testQUICStream, body func() io.ReadCloser) {
			var (
				bytesSent     int
				bytesReceived int
			)
			for _, step := range test.steps {
				switch step := step.(type) {
				case receiveHeaders:
					header := h.Clone()
					if step.contentLength != -1 {
						header["content-length"] = []string{
							fmt.Sprint(step.contentLength),
						}
					}
					st.writeHeaders(header)
				case receiveDataHeader:
					t.Logf("receive DATA frame header: size=%v", step.size)
					st.writeVarint(int64(frameTypeData))
					st.writeVarint(step.size)
					st.Flush()
				case receiveData:
					t.Logf("receive DATA frame content: size=%v", step.size)
					for range step.size {
						st.WriteByte(byte(bytesSent))
						bytesSent++
					}
					st.Flush()
				case receiveTrailers:
					st.writeHeaders(http.Header{
						"x-trailer": []string{"trailer"},
					})
				case receiveFrame:
					st.writeVarint(int64(step.ftype))
					st.writeVarint(int64(len(step.data)))
					st.Write(step.data)
					st.Flush()
				case receiveEOF:
					t.Logf("receive EOF on request stream")
					st.CloseWrite()
				case wantBody:
					t.Logf("read %v bytes from response body", step.size)
					want := make([]byte, step.size)
					for i := range want {
						want[i] = byte(bytesReceived)
						bytesReceived++
					}
					got := make([]byte, step.size)
					n, err := body().Read(got)
					got = got[:n]
					if !bytes.Equal(got, want) {
						t.Errorf("resp.Body.Read:")
						t.Errorf("  got:  {%x}", got)
						t.Fatalf("  want: {%x}", want)
					}
					if err != nil {
						if step.eof && err == io.EOF {
							continue
						}
						t.Fatalf("resp.Body.Read: unexpected error %v", err)
					}
					if step.eof {
						if n, err := body().Read([]byte{0}); n != 0 || err != io.EOF {
							t.Fatalf("resp.Body.Read() = %v, %v; want io.EOF", n, err)
						}
					}
				case wantError:
					n, err := body().Read([]byte{0})
					if n != 0 || err == nil || err == io.EOF {
						t.Fatalf("resp.Body.Read() = %v, %v; want error", n, err)
					}
					if step.err != nil && !errors.Is(step.err, err) {
						t.Fatalf("resp.Body.Read() = %v, %v; want %v error", n, err, step.err)
					}
				case closeBody:
					if err := body().Close(); err != nil {
						t.Fatalf("resp.Body.Close() = %v, want nil", err)
					}
				default:
					t.Fatalf("unknown test step %T", step)
				}
			}

		}

		synctestSubtest(t, test.name+"/client", func(t *testing.T) {
			tc := newTestClientConn(t)
			tc.greet()

			req, _ := http.NewRequest("GET", "https://example.tld/", nil)
			rt := tc.roundTrip(req)
			st := tc.wantStream(streamTypeRequest)
			st.wantHeaders(nil)

			header := http.Header{
				":status": []string{"200"},
			}
			runTest(t, header, st, func() io.ReadCloser {
				return rt.response().Body
			})
		})
	}
}
