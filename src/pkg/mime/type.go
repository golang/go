// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The mime package translates file name extensions to MIME types.
// It consults the local system's mime.types file, which must be installed
// under one of these names:
//
//   /etc/mime.types
//   /etc/apache2/mime.types
//   /etc/apache/mime.types
//
package mime

import (
	"bufio"
	"once"
	"os"
	"strings"
)

var typeFiles = []string{
	"/etc/mime.types",
	"/etc/apache2/mime.types",
	"/etc/apache/mime.types",
}

var mimeTypes = map[string]string{
	".css":  "text/css",
	".gif":  "image/gif",
	".htm":  "text/html; charset=utf-8",
	".html": "text/html; charset=utf-8",
	".jpg":  "image/jpeg",
	".js":   "application/x-javascript",
	".pdf":  "application/pdf",
	".png":  "image/png",
	".xml":  "text/xml; charset=utf-8",
}

func loadMimeFile(filename string) {
	f, err := os.Open(filename, os.O_RDONLY, 0666)
	if err != nil {
		return
	}

	reader := bufio.NewReader(f)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			f.Close()
			return
		}
		fields := strings.Fields(line)
		if len(fields) <= 1 || fields[0][0] == '#' {
			continue
		}
		typename := fields[0]
		if strings.HasPrefix(typename, "text/") {
			typename += "; charset=utf-8"
		}
		for _, ext := range fields[1:] {
			if ext[0] == '#' {
				break
			}
			mimeTypes["."+ext] = typename
		}
	}
}

func initMime() {
	for _, filename := range typeFiles {
		loadMimeFile(filename)
	}
}

// TypeByExtension returns the MIME type associated with the file extension ext.
// The extension ext should begin with a leading dot, as in ".html".
// When ext has no associated type, TypeByExtension returns "".
func TypeByExtension(ext string) string {
	once.Do(initMime)
	typ, _ := mimeTypes[ext]
	return typ
}
