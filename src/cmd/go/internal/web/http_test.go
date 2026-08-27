// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package web

import (
	"bytes"
	"compress/gzip"
	cryptorand "crypto/rand"
	"errors"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"testing"
	"testing/synctest"
	"time"
)

func TestUserAgent(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(r.UserAgent()))
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal("parse httptest url:", err)
	}
	res, err := Get(Insecure, u)
	if err != nil {
		t.Error("http get:", err)
	}
	defer res.Body.Close()
	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Error("read response body:", err)
	}
	gotUserAgent := string(bytes.TrimSpace(b))
	if gotUserAgent != userAgent {
		t.Errorf("User-Agent: %s, want %s", gotUserAgent, userAgent)
	}
}

func TestGoGet1(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/redirect-a":
			goGet := r.URL.Query().Get("go-get")
			if goGet != "1" {
				t.Errorf("missing go-get=1 on initial request: %s", r.URL.String())
			}
			http.Redirect(w, r, "/redirect-b", http.StatusFound)
		case "/redirect-b":
			goGet := r.URL.Query().Get("go-get")
			if goGet != "1" {
				t.Errorf("missing go-get=1 on redirected request: %s", r.URL.String())
			}
			http.Redirect(w, r, "/finish?other=param", http.StatusFound)
		case "/finish":
			goGet := r.URL.Query().Get("go-get")
			if goGet != "1" {
				t.Errorf("missing go-get=1 on final request: %s", r.URL.String())
			}
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal("parse httptest url:", err)
	}
	u.Path = "/redirect-a"
	u.RawQuery = "go-get=1"

	res, err := Get(Insecure, u)
	if err != nil {
		t.Fatalf("http get: %v", err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Errorf("http status != 200: %v", res.Status)
	}
}

var (
	testData     [1 << 20]byte
	gzipTestData []byte
)

func init() {
	cryptorand.Read(testData[:])

	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	zw.Write(testData[:])
	zw.Close()
	gzipTestData = buf.Bytes()
}

func setTestTransport(t *testing.T, client *http.Client) {
	insecureTransport := impatientInsecureHTTPClient.Transport
	secureTransport := securityPreservingDefaultClient.Transport
	t.Cleanup(func() {
		impatientInsecureHTTPClient.Transport = insecureTransport
		securityPreservingDefaultClient.Transport = secureTransport
	})
	impatientInsecureHTTPClient.Transport = client.Transport
	securityPreservingDefaultClient.Transport = client.Transport
}

func TestRetryResumed(t *testing.T) {
	type attempt struct {
		wantHeader http.Header

		respStatus int
		respHeader http.Header
		respBytes  []byte
	}
	rangeRequestHeader := func(ifHeader, rangeHeader string) http.Header {
		return {
			"If-Range": {ifHeader},
			"Range":    {rangeHeader},
		}
	}
	for _, test := range []struct {
		name      string
		wantData  []byte
		wantError bool
		attempts  []attempt
	}{{
		name:     "successful resume with 206 response",
		wantData: testData[:1000],
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":           {`"etag-123"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:500],
		}, {
			wantHeader: rangeRequestHeader(`"etag-123"`, "bytes=500-"),
			respStatus: 206, // Partial Content
			respHeader: {
				"ETag":          {`"etag-123"`},
				"Content-Range": {"bytes 500-999/1000"},
			},
			respBytes: testData[500:1000],
		}},
	}, {
		name:     "successful resume with oddly-cased 206 response",
		wantData: testData[:1000],
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":           {`"etag-123"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:500],
		}, {
			wantHeader: rangeRequestHeader(`"etag-123"`, "bytes=500-"),
			respStatus: 206, // Partial Content
			respHeader: {
				"ETag":          {`"etag-123"`},
				"Content-Range": {"bYtEs 500-999/1000"},
			},
			respBytes: testData[500:1000],
		}},
	}, {
		name:      "no resume with 200 response",
		wantError: true,
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":           {`"etag-123"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:500],
		}, {
			wantHeader: rangeRequestHeader(`"etag-123"`, "bytes=500-"),
			respStatus: 200,
			respHeader: {
				"ETag":           {`"etag-123"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:1000],
		}},
	}, {
		name:      "no resume after reading no bytes at start",
		wantError: true,
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":           {`"etag-123"`},
				"Content-Length": {"1000"},
			},
			respBytes: nil, // immediate failure
		}},
	}, {
		name:      "no resume after reading no bytes after resume",
		wantError: true,
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":           {`"etag-123"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:500],
		}, {
			wantHeader: rangeRequestHeader(`"etag-123"`, "bytes=500-"),
			respStatus: 206, // Partial Content
			respHeader: {
				"ETag":           {`"etag-123"`},
				"Content-Length": {"1000"},
			},
			respBytes: nil, // immediate failure
		}},
	}, {
		name:      "etag changed after resume",
		wantError: true,
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":           {`"etag-original"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:100],
		}, {
			wantHeader: rangeRequestHeader(`"etag-original"`, "bytes=100-"),
			respStatus: 206, // Partial Content
			respHeader: {
				"ETag":           {`"etag-changed"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:1000],
		}},
	}, {
		name:      "no resume with no identifier",
		wantError: true,
		attempts: {{
			respStatus: 200,
			respHeader: {
				"Content-Length": {"1000"},
			},
			respBytes: testData[:100],
		}},
	}, {
		name:      "no resume with weak etag",
		wantError: true,
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":           {`W/"tag"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:100],
		}},
	}, {
		name:     "successful resume with last modified",
		wantData: testData[:1000],
		attempts: {{
			respStatus: 200,
			respHeader: {
				"Last-Modified":  {"Sat, 01 Jan 2000 00:00:00 GMT"},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:100],
		}, {
			wantHeader: rangeRequestHeader("Sat, 01 Jan 2000 00:00:00 GMT", "bytes=100-"),
			respStatus: 206, // Partial Content
			respHeader: {
				"Last-Modified": {"Sat, 01 Jan 2000 00:00:00 GMT"},
				"Content-Range": {"bytes 100-999/1000"},
			},
			respBytes: testData[100:1000],
		}},
	}, {
		name:      "too many restarts",
		wantError: true,
		attempts: {{
			// attempt 1
			respStatus: 200,
			respHeader: {
				"ETag":           {`"etag-123"`},
				"Content-Length": {"1000"},
			},
			respBytes: testData[:100],
		}, {
			// attempt 2
			wantHeader: rangeRequestHeader(`"etag-123"`, "bytes=100-"),
			respStatus: 206, // Partial Content
			respHeader: {
				"ETag":          {`"etag-123"`},
				"Content-Range": {"bytes 100-999/1000"},
			},
			respBytes: testData[100:200],
		}, {
			// attempt 3
			wantHeader: rangeRequestHeader(`"etag-123"`, "bytes=200-"),
			respStatus: 206, // Partial Content
			respHeader: {
				"ETag":          {`"etag-123"`},
				"Content-Range": {"bytes 200-999/1000"},
			},
			respBytes: testData[200:300],
		}},
	}, {
		name:     "gzipped content success",
		wantData: testData[:],
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":             {`"etag-123"`},
				"Content-Encoding": {"gzip"},
			},
			respBytes: gzipTestData,
		}},
	}, {
		name:      "gzipped content cannot resume",
		wantError: true,
		attempts: {{
			respStatus: 200,
			respHeader: {
				"ETag":             {`"etag-123"`},
				"Content-Encoding": {"gzip"},
			},
			respBytes: gzipTestData[:100],
		}},
	}} {
		synctest.Subtest(t, test.name, func(t *testing.T) {
			attempts := 0
			ts := httptest.NewTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if attempts >= len(test.attempts) {
					attempts++
					w.WriteHeader(http.StatusNotAcceptable)
					return
				}
				at := test.attempts[attempts]
				attempts++

				// If the test didn't specify these headers,
				// they should not be present.
				wantHeader := http.Header{
					"Range":    nil,
					"If-Range": nil,
				}
				maps.Copy(wantHeader, at.wantHeader)
				for k, want := range wantHeader {
					if got := req.Header[k]; !slices.Equal(got, want) {
						t.Errorf("request %v: %v = %q, want %q", attempts, k, got, want)
					}
				}
				maps.Copy(w.Header(), at.respHeader)
				w.WriteHeader(at.respStatus)
				w.Write(at.respBytes)
			}))
			setTestTransport(t, ts.Client())

			gotBytes, err := GetBytes(url.MustParse("http://go.dev/"))
			if got, want := attempts, len(test.attempts); got != want {
				t.Errorf("GetBytes made %v requests, want %v", got, want)
				t.Errorf("read %v bytes, err=%v", len(gotBytes), err)
			}
			if err != nil {
				if !test.wantError {
					t.Errorf("GetBytes failed: %v", err)
				}
				return
			}
			if test.wantError {
				t.Fatalf("GetBytes succeeded (want error)")
			}
			if !bytes.Equal(gotBytes, test.wantData) {
				t.Errorf("download data mismatch")
			}
		})
	}
}

// TestRetryResumeServeContent exercises the resume-broken-download
// path using ServeContent, which is our own implementation of serving range requests.
// TestRetryResumed above checks that our resumption behavior is what we expect;
// this test checks that what we expect works (at least with ServeContent).
func TestRetryResumeServeContent(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		errsAt := []int64{100, 500}
		content := &retryContent{
			data:   testData[:],
			errsAt: errsAt,
		}
		modTime := time.Now()
		attempts := 0
		ts := httptest.NewTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			attempts++
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Header().Set("ETag", `"etag-123"`)
			http.ServeContent(w, req, "content", modTime, content)
		}))
		setTestTransport(t, ts.Client())

		got, err := GetBytes(url.MustParse("https://go.dev/"))
		if err != nil {
			t.Errorf("GetBytes failed: %v", err)
		}
		if !bytes.Equal(got, testData[:]) {
			t.Errorf("download data mismatch")
		}
		if got, want := attempts, 1+len(errsAt); got != want {
			t.Errorf("%v attempts, want %v", got, want)
		}
	})
}

// retryContent is an io.ReadSeeker for use with http.ServeContent.
//
// It returns an error the first time it reaches each location in errsAt.
// For example, if errsAt = [1, 10, 5]:
//   - the first read to reach byte 1 fails; then
//   - the next read to reach byte 10 fails; then
//   - the next read to reach byte 5 fails.
type retryContent struct {
	data   []byte
	errsAt []int64
	off    int64
}

func (c *retryContent) Read(p []byte) (n int, err error) {
	end := c.off + int64(len(p))
	if len(c.errsAt) > 0 && c.errsAt[0] >= c.off && c.errsAt[0] <= end {
		size := c.errsAt[0] - c.off
		p = p[:size]
		err = errors.New("read error")
		c.errsAt = c.errsAt[1:]
	}
	n = copy(p, c.data[c.off:])
	c.off += int64(n)
	return n, err
}

func (c *retryContent) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case io.SeekStart:
		c.off = offset
	case io.SeekCurrent:
		c.off += offset
	case io.SeekEnd:
		c.off = int64(len(c.data)) + offset
	}
	if c.off < 0 || c.off > int64(len(c.data)) {
		c.off = 0
		return 0, errors.New("invalid seek")
	}
	return c.off, nil
}
