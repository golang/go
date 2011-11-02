// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mime implements parts of the MIME spec.
package mime

import (
	"bufio"
	"fmt"
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
	".css":  "text/css; charset=utf-8",
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
	f, err := os.Open(filename)
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
		mimeType := fields[0]
		for _, ext := range fields[1:] {
			if ext[0] == '#' {
				break
			}
			setExtensionType("."+ext, mimeType)
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
//
// Text types have the charset parameter set to "utf-8" by default.
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
func AddExtensionType(ext, typ string) error {
	if ext == "" || ext[0] != '.' {
		return fmt.Errorf(`mime: extension "%s" misses dot`, ext)
	}
	once.Do(initMime)
	return setExtensionType(ext, typ)
}

func setExtensionType(extension, mimeType string) error {
	full, param, err := ParseMediaType(mimeType)
	if err != nil {
		return err
	}
	if split := strings.Index(full, "/"); split < 0 {
		return fmt.Errorf(`mime: malformed MIME type "%s"`, mimeType)
	} else {
		main := full[:split]
		sub := full[split+1:]
		if main == "text" && param["charset"] == "" {
			param["charset"] = "utf-8"
		}
		mimeType = FormatMediaType(main, sub, param)
	}

	mimeLock.Lock()
	mimeTypes[extension] = mimeType
	mimeLock.Unlock()
	return nil
}
