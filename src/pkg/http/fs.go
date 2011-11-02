// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP file system request handler

package http

import (
	"errors"
	"fmt"
	"io"
	"mime"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"utf8"
)

// A Dir implements http.FileSystem using the native file
// system restricted to a specific directory tree.
type Dir string

func (d Dir) Open(name string) (File, error) {
	if filepath.Separator != '/' && strings.IndexRune(name, filepath.Separator) >= 0 {
		return nil, errors.New("http: invalid character in file path")
	}
	f, err := os.Open(filepath.Join(string(d), filepath.FromSlash(path.Clean("/"+name))))
	if err != nil {
		return nil, err
	}
	return f, nil
}

// A FileSystem implements access to a collection of named files.
// The elements in a file path are separated by slash ('/', U+002F)
// characters, regardless of host operating system convention.
type FileSystem interface {
	Open(name string) (File, error)
}

// A File is returned by a FileSystem's Open method and can be
// served by the FileServer implementation.
type File interface {
	Close() error
	Stat() (*os.FileInfo, error)
	Readdir(count int) ([]os.FileInfo, error)
	Read([]byte) (int, error)
	Seek(offset int64, whence int) (int64, error)
}

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

func dirList(w ResponseWriter, f File) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
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

// name is '/'-separated, not filepath.Separator.
func serveFile(w ResponseWriter, r *Request, fs FileSystem, name string, redirect bool) {
	const indexPage = "/index.html"

	// redirect .../index.html to .../
	// can't use Redirect() because that would make the path absolute,
	// which would be a problem running under StripPrefix
	if strings.HasSuffix(r.URL.Path, indexPage) {
		localRedirect(w, r, "./")
		return
	}

	f, err := fs.Open(name)
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
				localRedirect(w, r, path.Base(url)+"/")
				return
			}
		} else {
			if url[len(url)-1] == '/' {
				localRedirect(w, r, "../"+path.Base(url))
				return
			}
		}
	}

	if t, _ := time.Parse(TimeFormat, r.Header.Get("If-Modified-Since")); t != nil && d.Mtime_ns/1e9 <= t.Seconds() {
		w.WriteHeader(StatusNotModified)
		return
	}
	w.Header().Set("Last-Modified", time.SecondsToUTC(d.Mtime_ns/1e9).Format(TimeFormat))

	// use contents of index.html for directory, if present
	if d.IsDirectory() {
		index := name + indexPage
		ff, err := fs.Open(index)
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

	// If Content-Type isn't set, use the file's extension to find it.
	if w.Header().Get("Content-Type") == "" {
		ctype := mime.TypeByExtension(filepath.Ext(name))
		if ctype == "" {
			// read a chunk to decide between utf-8 text and binary
			var buf [1024]byte
			n, _ := io.ReadFull(f, buf[:])
			b := buf[:n]
			if isText(b) {
				ctype = "text/plain; charset=utf-8"
			} else {
				// generic binary
				ctype = "application/octet-stream"
			}
			f.Seek(0, os.SEEK_SET) // rewind to output whole file
		}
		w.Header().Set("Content-Type", ctype)
	}

	// handle Content-Range header.
	// TODO(adg): handle multiple ranges
	ranges, err := parseRange(r.Header.Get("Range"), size)
	if err == nil && len(ranges) > 1 {
		err = errors.New("multiple ranges not supported")
	}
	if err != nil {
		Error(w, err.Error(), StatusRequestedRangeNotSatisfiable)
		return
	}
	if len(ranges) == 1 {
		ra := ranges[0]
		if _, err := f.Seek(ra.start, os.SEEK_SET); err != nil {
			Error(w, err.Error(), StatusRequestedRangeNotSatisfiable)
			return
		}
		size = ra.length
		code = StatusPartialContent
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", ra.start, ra.start+ra.length-1, d.Size))
	}

	w.Header().Set("Accept-Ranges", "bytes")
	if w.Header().Get("Content-Encoding") == "" {
		w.Header().Set("Content-Length", strconv.Itoa64(size))
	}

	w.WriteHeader(code)

	if r.Method != "HEAD" {
		io.CopyN(w, f, size)
	}
}

// localRedirect gives a Moved Permanently response.
// It does not convert relative paths to absolute paths like Redirect does.
func localRedirect(w ResponseWriter, r *Request, newPath string) {
	if q := r.URL.RawQuery; q != "" {
		newPath += "?" + q
	}
	w.Header().Set("Location", newPath)
	w.WriteHeader(StatusMovedPermanently)
}

// ServeFile replies to the request with the contents of the named file or directory.
func ServeFile(w ResponseWriter, r *Request, name string) {
	dir, file := filepath.Split(name)
	serveFile(w, r, Dir(dir), file, false)
}

type fileHandler struct {
	root FileSystem
}

// FileServer returns a handler that serves HTTP requests
// with the contents of the file system rooted at root.
//
// To use the operating system's file system implementation,
// use http.Dir:
//
//     http.Handle("/", http.FileServer(http.Dir("/tmp")))
func FileServer(root FileSystem) Handler {
	return &fileHandler{root}
}

func (f *fileHandler) ServeHTTP(w ResponseWriter, r *Request) {
	upath := r.URL.Path
	if !strings.HasPrefix(upath, "/") {
		upath = "/" + upath
		r.URL.Path = upath
	}
	serveFile(w, r, f.root, path.Clean(upath), true)
}

// httpRange specifies the byte range to be sent to the client.
type httpRange struct {
	start, length int64
}

// parseRange parses a Range header string as per RFC 2616.
func parseRange(s string, size int64) ([]httpRange, error) {
	if s == "" {
		return nil, nil // header not present
	}
	const b = "bytes="
	if !strings.HasPrefix(s, b) {
		return nil, errors.New("invalid range")
	}
	var ranges []httpRange
	for _, ra := range strings.Split(s[len(b):], ",") {
		i := strings.Index(ra, "-")
		if i < 0 {
			return nil, errors.New("invalid range")
		}
		start, end := ra[:i], ra[i+1:]
		var r httpRange
		if start == "" {
			// If no start is specified, end specifies the
			// range start relative to the end of the file.
			i, err := strconv.Atoi64(end)
			if err != nil {
				return nil, errors.New("invalid range")
			}
			if i > size {
				i = size
			}
			r.start = size - i
			r.length = size - r.start
		} else {
			i, err := strconv.Atoi64(start)
			if err != nil || i > size || i < 0 {
				return nil, errors.New("invalid range")
			}
			r.start = i
			if end == "" {
				// If no end is specified, range extends to end of the file.
				r.length = size - r.start
			} else {
				i, err := strconv.Atoi64(end)
				if err != nil || r.start > i {
					return nil, errors.New("invalid range")
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
