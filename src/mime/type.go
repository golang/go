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
	mimeLock       sync.RWMutex      // guards following 3 maps
	mimeTypes      map[string]string // ".Z" => "application/x-compress"
	mimeTypesLower map[string]string // ".z" => "application/x-compress"

	// extensions maps from MIME type to list of lowercase file
	// extensions: "image/jpeg" => [".jpg", ".jpeg"]
	extensions map[string][]string
)

// setMimeTypes is used by initMime's non-test path, and by tests.
// The two maps must not be the same, or nil.
func setMimeTypes(lowerExt, mixExt map[string]string) {
	if lowerExt == nil || mixExt == nil {
		panic("nil map")
	}
	mimeTypesLower = lowerExt
	mimeTypes = mixExt
	extensions = invert(lowerExt)
}

var builtinTypesLower = map[string]string{
	".css":  "text/css; charset=utf-8",
	".gif":  "image/gif",
	".htm":  "text/html; charset=utf-8",
	".html": "text/html; charset=utf-8",
	".jpg":  "image/jpeg",
	".js":   "application/x-javascript",
	".pdf":  "application/pdf",
	".png":  "image/png",
	".svg":  "image/svg+xml",
	".xml":  "text/xml; charset=utf-8",
}

func clone(m map[string]string) map[string]string {
	m2 := make(map[string]string, len(m))
	for k, v := range m {
		m2[k] = v
		if strings.ToLower(k) != k {
			panic("keys in builtinTypesLower must be lowercase")
		}
	}
	return m2
}

func invert(m map[string]string) map[string][]string {
	m2 := make(map[string][]string, len(m))
	for k, v := range m {
		justType, _, err := ParseMediaType(v)
		if err != nil {
			panic(err)
		}
		m2[justType] = append(m2[justType], k)
	}
	return m2
}

var once sync.Once // guards initMime

var testInitMime, osInitMime func()

func initMime() {
	if fn := testInitMime; fn != nil {
		fn()
	} else {
		setMimeTypes(builtinTypesLower, clone(builtinTypesLower))
		osInitMime()
	}
}

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
// On Windows, MIME types are extracted from the registry.
//
// Text types have the charset parameter set to "utf-8" by default.
func TypeByExtension(ext string) string {
	once.Do(initMime)
	mimeLock.RLock()
	defer mimeLock.RUnlock()

	// Case-sensitive lookup.
	if v := mimeTypes[ext]; v != "" {
		return v
	}

	// Case-insensitive lookup.
	// Optimistically assume a short ASCII extension and be
	// allocation-free in that case.
	var buf [10]byte
	lower := buf[:0]
	const utf8RuneSelf = 0x80 // from utf8 package, but not importing it.
	for i := 0; i < len(ext); i++ {
		c := ext[i]
		if c >= utf8RuneSelf {
			// Slow path.
			return mimeTypesLower[strings.ToLower(ext)]
		}
		if 'A' <= c && c <= 'Z' {
			lower = append(lower, c+('a'-'A'))
		} else {
			lower = append(lower, c)
		}
	}
	// The conversion from []byte to string doesn't allocate in
	// a map lookup.
	return mimeTypesLower[string(lower)]
}

// ExtensionsByType returns the extensions known to be associated with the MIME
// type typ. The returned extensions will each begin with a leading dot, as in
// ".html". When typ has no associated extensions, ExtensionsByType returns an
// nil slice.
func ExtensionsByType(typ string) ([]string, error) {
	justType, _, err := ParseMediaType(typ)
	if err != nil {
		return nil, err
	}

	once.Do(initMime)
	mimeLock.RLock()
	defer mimeLock.RUnlock()
	s, ok := extensions[justType]
	if !ok {
		return nil, nil
	}
	return append([]string{}, s...), nil
}

// AddExtensionType sets the MIME type associated with
// the extension ext to typ. The extension should begin with
// a leading dot, as in ".html".
func AddExtensionType(ext, typ string) error {
	if !strings.HasPrefix(ext, ".") {
		return fmt.Errorf("mime: extension %q missing leading dot", ext)
	}
	once.Do(initMime)
	return setExtensionType(ext, typ)
}

func setExtensionType(extension, mimeType string) error {
	justType, param, err := ParseMediaType(mimeType)
	if err != nil {
		return err
	}
	if strings.HasPrefix(mimeType, "text/") && param["charset"] == "" {
		param["charset"] = "utf-8"
		mimeType = FormatMediaType(mimeType, param)
	}
	extLower := strings.ToLower(extension)

	mimeLock.Lock()
	defer mimeLock.Unlock()
	mimeTypes[extension] = mimeType
	mimeTypesLower[extLower] = mimeType
	for _, v := range extensions[justType] {
		if v == extLower {
			return nil
		}
	}
	extensions[justType] = append(extensions[justType], extLower)
	return nil
}
