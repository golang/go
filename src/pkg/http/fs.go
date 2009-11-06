// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP file system request handler

package http

import (
	"fmt";
	"io";
	"os";
	"path";
	"strings";
	"utf8";
)

// TODO this should be in a mime package somewhere
var contentByExt = map[string]string{
	".css": "text/css",
	".gif": "image/gif",
	".html": "text/html; charset=utf-8",
	".jpg": "image/jpeg",
	".js": "application/x-javascript",
	".pdf": "application/pdf",
	".png": "image/png",
}

// Heuristic: b is text if it is valid UTF-8 and doesn't
// contain any unprintable ASCII or Unicode characters.
func isText(b []byte) bool {
	for len(b) > 0 && utf8.FullRune(b) {
		rune, size := utf8.DecodeRune(b);
		if size == 1 && rune == utf8.RuneError {
			// decoding error
			return false;
		}
		if 0x80 <= rune && rune <= 0x9F {
			return false;
		}
		if rune < ' ' {
			switch rune {
			case '\n', '\r', '\t':
				// okay
			default:
				// binary garbage
				return false;
			}
		}
		b = b[size:len(b)];
	}
	return true;
}

func dirList(c *Conn, f *os.File) {
	fmt.Fprintf(c, "<pre>\n");
	for {
		dirs, err := f.Readdir(100);
		if err != nil || len(dirs) == 0 {
			break;
		}
		for _, d := range dirs {
			name := d.Name;
			if d.IsDirectory() {
				name += "/";
			}
			// TODO htmlescape
			fmt.Fprintf(c, "<a href=\"%s\">%s</a>\n", name, name);
		}
	}
	fmt.Fprintf(c, "</pre>\n");
}


func serveFileInternal(c *Conn, r *Request, name string, redirect bool) {
	const indexPage = "/index.html";

	// redirect to strip off any index.html
	n := len(name)-len(indexPage);
	if n >= 0 && name[n:len(name)] == indexPage {
		Redirect(c, name[0 : n+1], StatusMovedPermanently);
		return;
	}

	f, err := os.Open(name, os.O_RDONLY, 0);
	if err != nil {
		// TODO expose actual error?
		NotFound(c, r);
		return;
	}
	defer f.Close();

	d, err1 := f.Stat();
	if err1 != nil {
		// TODO expose actual error?
		NotFound(c, r);
		return;
	}

	if redirect {
		// redirect to canonical path: / at end of directory url
		// r.Url.Path always begins with /
		url := r.Url.Path;
		if d.IsDirectory() {
			if url[len(url)-1] != '/' {
				Redirect(c, url+"/", StatusMovedPermanently);
				return;
			}
		} else {
			if url[len(url)-1] == '/' {
				Redirect(c, url[0 : len(url)-1], StatusMovedPermanently);
				return;
			}
		}
	}

	// use contents of index.html for directory, if present
	if d.IsDirectory() {
		index := name + indexPage;
		ff, err := os.Open(index, os.O_RDONLY, 0);
		if err == nil {
			defer ff.Close();
			dd, err := ff.Stat();
			if err == nil {
				name = index;
				d = dd;
				f = ff;
			}
		}
	}

	if d.IsDirectory() {
		dirList(c, f);
		return;
	}

	// serve file
	// use extension to find content type.
	ext := path.Ext(name);
	if ctype, ok := contentByExt[ext]; ok {
		c.SetHeader("Content-Type", ctype);
	} else {
		// read first chunk to decide between utf-8 text and binary
		var buf [1024]byte;
		n, _ := io.ReadFull(f, &buf);
		b := buf[0:n];
		if isText(b) {
			c.SetHeader("Content-Type", "text-plain; charset=utf-8");
		} else {
			c.SetHeader("Content-Type", "application/octet-stream");	// generic binary
		}
		c.Write(b);
	}
	io.Copy(c, f);
}

// ServeFile replies to the request with the contents of the named file or directory.
func ServeFile(c *Conn, r *Request, name string) {
	serveFileInternal(c, r, name, false);
}

type fileHandler struct {
	root	string;
	prefix	string;
}

// FileServer returns a handler that serves HTTP requests
// with the contents of the file system rooted at root.
// It strips prefix from the incoming requests before
// looking up the file name in the file system.
func FileServer(root, prefix string) Handler	{ return &fileHandler{root, prefix} }

func (f *fileHandler) ServeHTTP(c *Conn, r *Request) {
	path := r.Url.Path;
	if !strings.HasPrefix(path, f.prefix) {
		NotFound(c, r);
		return;
	}
	path = path[len(f.prefix):len(path)];
	serveFileInternal(c, r, f.root + "/" + path, true);
}
