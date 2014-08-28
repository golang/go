// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mime implements parts of the MIME spec.
package mime

import (
	"fmt"
	"strings"
	"sync"
)

var (
	mimeLock       sync.RWMutex
	mimeTypes      = map[string]string{}
	mimeTypesLower = map[string]string{}
)

func init() {
	mimeTypes := map[string]string{
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
	for ext, typ := range mimeTypes {
		AddExtensionType(ext, typ)
	}
}

var once sync.Once

// TypeByExtension returns the MIME type associated with the file extension ext.
// The extension ext should begin with a leading dot, as in ".html".
// When ext has no associated type, TypeByExtension returns "".
//
// Extensions are looked up first case-sensitively, then case-insensitively.
//
// The built-in table is small but on unix it is augmented by the local
// system's mime.types file(s) if available under one or more of these
// names:
//
//   /etc/mime.types
//   /etc/apache2/mime.types
//   /etc/apache/mime.types
//
// Windows system MIME types are extracted from registry.
//
// Text types have the charset parameter set to "utf-8" by default.
func TypeByExtension(ext string) string {
	once.Do(initMime)
	mimeLock.RLock()
	typename := mimeTypes[ext]
	mimeLock.RUnlock()
	if typename == "" {
		lower := strings.ToLower(ext)
		mimeLock.RLock()
		typename = mimeTypesLower[lower]
		mimeLock.RUnlock()
	}
	return typename
}

// AddExtensionType sets the MIME type associated with
// the extension ext to typ. The extension should begin with
// a leading dot, as in ".html".
func AddExtensionType(ext, typ string) error {
	if !strings.HasPrefix(ext, ".") {
		return fmt.Errorf(`mime: extension %q misses dot`, ext)
	}
	once.Do(initMime)
	return setExtensionType(ext, typ)
}

func setExtensionType(extension, mimeType string) error {
	_, param, err := ParseMediaType(mimeType)
	if err != nil {
		return err
	}
	if strings.HasPrefix(mimeType, "text/") && param["charset"] == "" {
		param["charset"] = "utf-8"
		mimeType = FormatMediaType(mimeType, param)
	}
	extLower := strings.ToLower(extension)

	mimeLock.Lock()
	mimeTypes[extension] = mimeType
	mimeTypesLower[extLower] = mimeType
	mimeLock.Unlock()
	return nil
}
