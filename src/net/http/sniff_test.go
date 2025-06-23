// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bytes"
	"fmt"
	"io"
	"log"
	. "net/http"
	"slices"
	"strconv"
	"strings"
	"testing"
)

var sniffTests = []struct {
	desc        string
	data        []byte
	contentType string
}{
	// Some nonsense.
	{"Empty", []byte{}, "text/plain; charset=utf-8"},
	{"Binary", []byte{1, 2, 3}, "application/octet-stream"},

	{"HTML document #1", []byte(`<HtMl><bOdY>blah blah blah</body></html>`), "text/html; charset=utf-8"},
	{"HTML document #2", []byte(`<HTML></HTML>`), "text/html; charset=utf-8"},
	{"HTML document #3 (leading whitespace)", []byte(`   <!DOCTYPE HTML>...`), "text/html; charset=utf-8"},
	{"HTML document #4 (leading CRLF)", []byte("\r\n<html>..."), "text/html; charset=utf-8"},

	{"Plain text", []byte(`This is not HTML. It has ☃ though.`), "text/plain; charset=utf-8"},

	{"XML", []byte("\n<?xml!"), "text/xml; charset=utf-8"},

	// Image types.
	{"Windows icon", []byte("\x00\x00\x01\x00"), "image/x-icon"},
	{"Windows cursor", []byte("\x00\x00\x02\x00"), "image/x-icon"},
	{"BMP image", []byte("BM..."), "image/bmp"},
	{"GIF 87a", []byte(`GIF87a`), "image/gif"},
	{"GIF 89a", []byte(`GIF89a...`), "image/gif"},
	{"WEBP image", []byte("RIFF\x00\x00\x00\x00WEBPVP"), "image/webp"},
	{"PNG image", []byte("\x89PNG\x0D\x0A\x1A\x0A"), "image/png"},
	{"JPEG image", []byte("\xFF\xD8\xFF"), "image/jpeg"},

	// Audio types.
	{"MIDI audio", []byte("MThd\x00\x00\x00\x06\x00\x01"), "audio/midi"},
	{"MP3 audio/MPEG audio", []byte("ID3\x03\x00\x00\x00\x00\x0f"), "audio/mpeg"},
	{"WAV audio #1", []byte("RIFFb\xb8\x00\x00WAVEfmt \x12\x00\x00\x00\x06"), "audio/wave"},
	{"WAV audio #2", []byte("RIFF,\x00\x00\x00WAVEfmt \x12\x00\x00\x00\x06"), "audio/wave"},
	{"AIFF audio #1", []byte("FORM\x00\x00\x00\x00AIFFCOMM\x00\x00\x00\x12\x00\x01\x00\x00\x57\x55\x00\x10\x40\x0d\xf3\x34"), "audio/aiff"},

	{"OGG audio", []byte("OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x7e\x46\x00\x00\x00\x00\x00\x00\x1f\xf6\xb4\xfc\x01\x1e\x01\x76\x6f\x72"), "application/ogg"},
	{"Must not match OGG", []byte("owow\x00"), "application/octet-stream"},
	{"Must not match OGG", []byte("oooS\x00"), "application/octet-stream"},
	{"Must not match OGG", []byte("oggS\x00"), "application/octet-stream"},

	// Video types.
	{"MP4 video", []byte("\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom<\x06t\xbfmdat"), "video/mp4"},
	{"AVI video #1", []byte("RIFF,O\n\x00AVI LISTÀ"), "video/avi"},
	{"AVI video #2", []byte("RIFF,\n\x00\x00AVI LISTÀ"), "video/avi"},

	// Font types.
	// {"MS.FontObject", []byte("\x00\x00")},
	{"TTF sample  I", []byte("\x00\x01\x00\x00\x00\x17\x01\x00\x00\x04\x01\x60\x4f"), "font/ttf"},
	{"TTF sample II", []byte("\x00\x01\x00\x00\x00\x0e\x00\x80\x00\x03\x00\x60\x46"), "font/ttf"},

	{"OTTO sample  I", []byte("\x4f\x54\x54\x4f\x00\x0e\x00\x80\x00\x03\x00\x60\x42\x41\x53\x45"), "font/otf"},

	{"woff sample  I", []byte("\x77\x4f\x46\x46\x00\x01\x00\x00\x00\x00\x30\x54\x00\x0d\x00\x00"), "font/woff"},
	{"woff2 sample", []byte("\x77\x4f\x46\x32\x00\x01\x00\x00\x00"), "font/woff2"},
	{"wasm sample", []byte("\x00\x61\x73\x6d\x01\x00"), "application/wasm"},

	// Archive types
	{"RAR v1.5-v4.0", []byte("Rar!\x1A\x07\x00"), "application/x-rar-compressed"},
	{"RAR v5+", []byte("Rar!\x1A\x07\x01\x00"), "application/x-rar-compressed"},
	{"Incorrect RAR v1.5-v4.0", []byte("Rar \x1A\x07\x00"), "application/octet-stream"},
	{"Incorrect RAR v5+", []byte("Rar \x1A\x07\x01\x00"), "application/octet-stream"},
}

func TestDetectContentType(t *testing.T) {
	for _, tt := range sniffTests {
		ct := DetectContentType(tt.data)
		if ct != tt.contentType {
			t.Errorf("%v: DetectContentType = %q, want %q", tt.desc, ct, tt.contentType)
		}
	}
}

func TestServerContentTypeSniff(t *testing.T) { run(t, testServerContentTypeSniff) }
func testServerContentTypeSniff(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		i, _ := strconv.Atoi(r.FormValue("i"))
		tt := sniffTests[i]
		n, err := w.Write(tt.data)
		if n != len(tt.data) || err != nil {
			log.Fatalf("%v: Write(%q) = %v, %v want %d, nil", tt.desc, tt.data, n, err, len(tt.data))
		}
	}))
	defer cst.close()

	for i, tt := range sniffTests {
		resp, err := cst.c.Get(cst.ts.URL + "/?i=" + strconv.Itoa(i))
		if err != nil {
			t.Errorf("%v: %v", tt.desc, err)
			continue
		}
		// DetectContentType is defined to return
		// text/plain; charset=utf-8 for an empty body,
		// but as of Go 1.10 the HTTP server has been changed
		// to return no content-type at all for an empty body.
		// Adjust the expectation here.
		wantContentType := tt.contentType
		if len(tt.data) == 0 {
			wantContentType = ""
		}
		if ct := resp.Header.Get("Content-Type"); ct != wantContentType {
			t.Errorf("%v: Content-Type = %q, want %q", tt.desc, ct, wantContentType)
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%v: reading body: %v", tt.desc, err)
		} else if !bytes.Equal(data, tt.data) {
			t.Errorf("%v: data is %q, want %q", tt.desc, data, tt.data)
		}
		resp.Body.Close()
	}
}

// Issue 5953: shouldn't sniff if the handler set a Content-Type header,
// even if it's the empty string.
func TestServerIssue5953(t *testing.T) { run(t, testServerIssue5953) }
func testServerIssue5953(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header()["Content-Type"] = []string{""}
		fmt.Fprintf(w, "<html><head></head><body>hi</body></html>")
	}))

	resp, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	got := resp.Header["Content-Type"]
	want := []string{""}
	if !slices.Equal(got, want) {
		t.Errorf("Content-Type = %q; want %q", got, want)
	}
	resp.Body.Close()
}

type byteAtATimeReader struct {
	buf []byte
}

func (b *byteAtATimeReader) Read(p []byte) (n int, err error) {
	if len(p) < 1 {
		return 0, nil
	}
	if len(b.buf) == 0 {
		return 0, io.EOF
	}
	p[0] = b.buf[0]
	b.buf = b.buf[1:]
	return 1, nil
}

func TestContentTypeWithVariousSources(t *testing.T) { run(t, testContentTypeWithVariousSources) }
func testContentTypeWithVariousSources(t *testing.T, mode testMode) {
	const (
		input    = "\n<html>\n\t<head>\n"
		expected = "text/html; charset=utf-8"
	)

	for _, test := range []struct {
		name    string
		handler func(ResponseWriter, *Request)
	}{{
		name: "write",
		handler: func(w ResponseWriter, r *Request) {
			// Write the whole input at once.
			n, err := w.Write([]byte(input))
			if int(n) != len(input) || err != nil {
				t.Errorf("w.Write(%q) = %v, %v want %d, nil", input, n, err, len(input))
			}
		},
	}, {
		name: "write one byte at a time",
		handler: func(w ResponseWriter, r *Request) {
			// Write the input one byte at a time.
			buf := []byte(input)
			for i := range buf {
				n, err := w.Write(buf[i : i+1])
				if n != 1 || err != nil {
					t.Errorf("w.Write(%q) = %v, %v want 1, nil", input, n, err)
				}
			}
		},
	}, {
		name: "copy from Reader",
		handler: func(w ResponseWriter, r *Request) {
			// Use io.Copy from a plain Reader.
			type readerOnly struct{ io.Reader }
			buf := bytes.NewBuffer([]byte(input))
			n, err := io.Copy(w, readerOnly{buf})
			if int(n) != len(input) || err != nil {
				t.Errorf("io.Copy(w, %q) = %v, %v want %d, nil", input, n, err, len(input))
			}
		},
	}, {
		name: "copy from bytes.Buffer",
		handler: func(w ResponseWriter, r *Request) {
			// Use io.Copy from a bytes.Buffer to trigger ReadFrom.
			buf := bytes.NewBuffer([]byte(input))
			n, err := io.Copy(w, buf)
			if int(n) != len(input) || err != nil {
				t.Errorf("io.Copy(w, %q) = %v, %v want %d, nil", input, n, err, len(input))
			}
		},
	}, {
		name: "copy one byte at a time",
		handler: func(w ResponseWriter, r *Request) {
			// Use io.Copy from a Reader that returns one byte at a time.
			n, err := io.Copy(w, &byteAtATimeReader{[]byte(input)})
			if int(n) != len(input) || err != nil {
				t.Errorf("io.Copy(w, %q) = %v, %v want %d, nil", input, n, err, len(input))
			}
		},
	}} {
		t.Run(test.name, func(t *testing.T) {
			cst := newClientServerTest(t, mode, HandlerFunc(test.handler))

			resp, err := cst.c.Get(cst.ts.URL)
			if err != nil {
				t.Fatalf("Get: %v", err)
			}
			if ct := resp.Header.Get("Content-Type"); ct != expected {
				t.Errorf("Content-Type = %q, want %q", ct, expected)
			}
			if want, got := resp.Header.Get("Content-Length"), fmt.Sprint(len(input)); want != got {
				t.Errorf("Content-Length = %q, want %q", want, got)
			}
			data, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Errorf("reading body: %v", err)
			} else if !bytes.Equal(data, []byte(input)) {
				t.Errorf("data is %q, want %q", data, input)
			}
			resp.Body.Close()

		})

	}
}

func TestSniffWriteSize(t *testing.T) { run(t, testSniffWriteSize) }
func testSniffWriteSize(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		size, _ := strconv.Atoi(r.FormValue("size"))
		written, err := io.WriteString(w, strings.Repeat("a", size))
		if err != nil {
			t.Errorf("write of %d bytes: %v", size, err)
			return
		}
		if written != size {
			t.Errorf("write of %d bytes wrote %d bytes", size, written)
		}
	}))
	for _, size := range []int{0, 1, 200, 600, 999, 1000, 1023, 1024, 512 << 10, 1 << 20} {
		res, err := cst.c.Get(fmt.Sprintf("%s/?size=%d", cst.ts.URL, size))
		if err != nil {
			t.Fatalf("size %d: %v", size, err)
		}
		if _, err := io.Copy(io.Discard, res.Body); err != nil {
			t.Fatalf("size %d: io.Copy of body = %v", size, err)
		}
		if err := res.Body.Close(); err != nil {
			t.Fatalf("size %d: body Close = %v", size, err)
		}
	}
}
