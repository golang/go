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
		if 0x80 <= rune && rune <= 0x9F {
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
	// use extension to find content type.
	ext := path.Ext(name)
	if ctype := mime.TypeByExtension(ext); ctype != "" {
		w.SetHeader("Content-Type", ctype)
	} else {
		// read first chunk to decide between utf-8 text and binary
		var buf [1024]byte
		n, _ := io.ReadFull(f, buf[0:])
		b := buf[0:n]
		if isText(b) {
			w.SetHeader("Content-Type", "text-plain; charset=utf-8")
		} else {
			w.SetHeader("Content-Type", "application/octet-stream") // generic binary
		}
		w.Write(b)
	}
	io.Copy(w, f)
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
