// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The mime package implements parts of the MIME spec.
package mime

import (
	"bufio"
	"os"
	"strings"
	"sync"
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

var mimeLock sync.RWMutex

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

var once sync.Once

// TypeByExtension returns the MIME type associated with the file extension ext.
// The extension ext should begin with a leading dot, as in ".html".
// When ext has no associated type, TypeByExtension returns "".
//
// The built-in table is small but is is augmented by the local
// system's mime.types file(s) if available under one or more of these
// names:
//
//   /etc/mime.types
//   /etc/apache2/mime.types
//   /etc/apache/mime.types
func TypeByExtension(ext string) string {
	once.Do(initMime)
	mimeLock.RLock()
	typename := mimeTypes[ext]
	mimeLock.RUnlock()
	return typename
}

// AddExtensionType sets the MIME type associated with
// the extension ext to typ.  The extension should begin with
// a leading dot, as in ".html".
func AddExtensionType(ext, typ string) os.Error {
	once.Do(initMime)
	if len(ext) < 1 || ext[0] != '.' {
		return os.EINVAL
	}
	mimeLock.Lock()
	mimeTypes[ext] = typ
	mimeLock.Unlock()
	return nil
}
