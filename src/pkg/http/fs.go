// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP file system request handler

package http

import (
	"fmt"
	"io"
	"mime"
	"os"
	"path"
	"strconv"
	"strings"
	"time"
	"utf8"
)

// Heuristic: b is text if it is valid UTF-8 and doesn't
// contain any unprintable ASCII or Unicode characters.
func isText(b []byte) bool {
	for len(b) > 0 && utf8.FullRune(b) {
		rune, size := utf8.DecodeRune(b)
		if size == 1 && rune == utf8.RuneError {
			// decoding error
			return false
		}
		if 0x7F <= rune && rune <= 0x9F {
			return false
		}
		if rune < ' ' {
			switch rune {
			case '\n', '\r', '\t':
				// okay
			default:
				// binary garbage
				return false
			}
		}
		b = b[size:]
	}
	return true
}

func dirList(w ResponseWriter, f *os.File) {
	fmt.Fprintf(w, "<pre>\n")
	for {
		dirs, err := f.Readdir(100)
		if err != nil || len(dirs) == 0 {
			break
		}
		for _, d := range dirs {
			name := d.Name
			if d.IsDirectory() {
				name += "/"
			}
			// TODO htmlescape
			fmt.Fprintf(w, "<a href=\"%s\">%s</a>\n", name, name)
		}
	}
	fmt.Fprintf(w, "</pre>\n")
}

func serveFile(w ResponseWriter, r *Request, name string, redirect bool) {
	const indexPage = "/index.html"

	// redirect .../index.html to .../
	if strings.HasSuffix(r.URL.Path, indexPage) {
		Redirect(w, r, r.URL.Path[0:len(r.URL.Path)-len(indexPage)+1], StatusMovedPermanently)
		return
	}

	f, err := os.Open(name, os.O_RDONLY, 0)
	if err != nil {
		// TODO expose actual error?
		NotFound(w, r)
		return
	}
	defer f.Close()

	d, err1 := f.Stat()
	if err1 != nil {
		// TODO expose actual error?
		NotFound(w, r)
		return
	}

	if redirect {
		// redirect to canonical path: / at end of directory url
		// r.URL.Path always begins with /
		url := r.URL.Path
		if d.IsDirectory() {
			if url[len(url)-1] != '/' {
				Redirect(w, r, url+"/", StatusMovedPermanently)
				return
			}
		} else {
			if url[len(url)-1] == '/' {
				Redirect(w, r, url[0:len(url)-1], StatusMovedPermanently)
				return
			}
		}
	}

	if t, _ := time.Parse(TimeFormat, r.Header["If-Modified-Since"]); t != nil && d.Mtime_ns/1e9 <= t.Seconds() {
		w.WriteHeader(StatusNotModified)
		return
	}
	w.SetHeader("Last-Modified", time.SecondsToUTC(d.Mtime_ns/1e9).Format(TimeFormat))

	// use contents of index.html for directory, if present
	if d.IsDirectory() {
		index := name + indexPage
		ff, err := os.Open(index, os.O_RDONLY, 0)
		if err == nil {
			defer ff.Close()
			dd, err := ff.Stat()
			if err == nil {
				name = index
				d = dd
				f = ff
			}
		}
	}

	if d.IsDirectory() {
		dirList(w, f)
		return
	}

	// serve file
	size := d.Size
	code := StatusOK

	// use extension to find content type.
	ext := path.Ext(name)
	if ctype := mime.TypeByExtension(ext); ctype != "" {
		w.SetHeader("Content-Type", ctype)
	} else {
		// read first chunk to decide between utf-8 text and binary
		var buf [1024]byte
		n, _ := io.ReadFull(f, buf[:])
		b := buf[:n]
		if isText(b) {
			w.SetHeader("Content-Type", "text-plain; charset=utf-8")
		} else {
			w.SetHeader("Content-Type", "application/octet-stream") // generic binary
		}
		f.Seek(0, 0) // rewind to output whole file
	}

	// handle Content-Range header.
	// TODO(adg): handle multiple ranges
	ranges, err := parseRange(r.Header["Range"], size)
	if err != nil || len(ranges) > 1 {
		Error(w, err.String(), StatusRequestedRangeNotSatisfiable)
		return
	}
	if len(ranges) == 1 {
		ra := ranges[0]
		if _, err := f.Seek(ra.start, 0); err != nil {
			Error(w, err.String(), StatusRequestedRangeNotSatisfiable)
			return
		}
		size = ra.length
		code = StatusPartialContent
		w.SetHeader("Content-Range", fmt.Sprintf("bytes %d-%d/%d", ra.start, ra.start+ra.length-1, d.Size))
	}

	w.SetHeader("Accept-Ranges", "bytes")
	w.SetHeader("Content-Length", strconv.Itoa64(size))

	w.WriteHeader(code)

	if r.Method != "HEAD" {
		io.Copyn(w, f, size)
	}
}

// ServeFile replies to the request with the contents of the named file or directory.
func ServeFile(w ResponseWriter, r *Request, name string) {
	serveFile(w, r, name, false)
}

type fileHandler struct {
	root   string
	prefix string
}

// FileServer returns a handler that serves HTTP requests
// with the contents of the file system rooted at root.
// It strips prefix from the incoming requests before
// looking up the file name in the file system.
func FileServer(root, prefix string) Handler { return &fileHandler{root, prefix} }

func (f *fileHandler) ServeHTTP(w ResponseWriter, r *Request) {
	path := r.URL.Path
	if !strings.HasPrefix(path, f.prefix) {
		NotFound(w, r)
		return
	}
	path = path[len(f.prefix):]
	serveFile(w, r, f.root+"/"+path, true)
}

// httpRange specifies the byte range to be sent to the client.
type httpRange struct {
	start, length int64
}

// parseRange parses a Range header string as per RFC 2616.
func parseRange(s string, size int64) ([]httpRange, os.Error) {
	if s == "" {
		return nil, nil // header not present
	}
	const b = "bytes="
	if !strings.HasPrefix(s, b) {
		return nil, os.NewError("invalid range")
	}
	var ranges []httpRange
	for _, ra := range strings.Split(s[len(b):], ",", -1) {
		i := strings.Index(ra, "-")
		if i < 0 {
			return nil, os.NewError("invalid range")
		}
		start, end := ra[:i], ra[i+1:]
		var r httpRange
		if start == "" {
			// If no start is specified, end specifies the
			// range start relative to the end of the file.
			i, err := strconv.Atoi64(end)
			if err != nil {
				return nil, os.NewError("invalid range")
			}
			if i > size {
				i = size
			}
			r.start = size - i
			r.length = size - r.start
		} else {
			i, err := strconv.Atoi64(start)
			if err != nil || i > size || i < 0 {
				return nil, os.NewError("invalid range")
			}
			r.start = i
			if end == "" {
				// If no end is specified, range extends to end of the file.
				r.length = size - r.start
			} else {
				i, err := strconv.Atoi64(end)
				if err != nil || r.start > i {
					return nil, os.NewError("invalid range")
				}
				if i >= size {
					i = size - 1
				}
				r.length = i - r.start + 1
			}
		}
		ranges = append(ranges, r)
	}
	return ranges, nil
}
