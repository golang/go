// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mime implements parts of the MIME spec.
package mime

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

var (
	mimeTypes      sync.Map // map[string]string; ".Z" => "application/x-compress"
	mimeTypesLower sync.Map // map[string]string; ".z" => "application/x-compress"

	// extensions maps from MIME type to list of lowercase file
	// extensions: "image/jpeg" => [".jpg", ".jpeg"]
	extensionsMu sync.Mutex // Guards stores (but not loads) on extensions.
	extensions   sync.Map   // map[string][]string; slice values are append-only.
)

func clearSyncMap(m *sync.Map) {
	m.Range(func(k, _ any) bool {
		m.Delete(k)
		return true
	})
}

// setMimeTypes is used by initMime's non-test path, and by tests.
func setMimeTypes(lowerExt, mixExt map[string]string) {
	clearSyncMap(&mimeTypes)
	clearSyncMap(&mimeTypesLower)
	clearSyncMap(&extensions)

	for k, v := range lowerExt {
		mimeTypesLower.Store(k, v)
	}
	for k, v := range mixExt {
		mimeTypes.Store(k, v)
	}

	extensionsMu.Lock()
	defer extensionsMu.Unlock()
	for k, v := range lowerExt {
		justType, _, err := ParseMediaType(v)
		if err != nil {
			panic(err)
		}
		var exts []string
		if ei, ok := extensions.Load(justType); ok {
			exts = ei.([]string)
		}
		extensions.Store(justType, append(exts, k))
	}
}

var builtinTypesLower = map[string]string{
	".avif": "image/avif",
	".css":  "text/css; charset=utf-8",
	".gif":  "image/gif",
	".htm":  "text/html; charset=utf-8",
	".html": "text/html; charset=utf-8",
	".jpeg": "image/jpeg",
	".jpg":  "image/jpeg",
	".js":   "text/javascript; charset=utf-8",
	".json": "application/json",
	".mjs":  "text/javascript; charset=utf-8",
	".pdf":  "application/pdf",
	".png":  "image/png",
	".svg":  "image/svg+xml",
	".wasm": "application/wasm",
	".webp": "image/webp",
	".xml":  "text/xml; charset=utf-8",
}

var once sync.Once // guards initMime

var testInitMime, osInitMime func()

func initMime() {
	if fn := testInitMime; fn != nil {
		fn()
	} else {
		setMimeTypes(builtinTypesLower, builtinTypesLower)
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
// system's MIME-info database or mime.types file(s) if available under one or
// more of these names:
//
//	/usr/local/share/mime/globs2
//	/usr/share/mime/globs2
//	/etc/mime.types
//	/etc/apache2/mime.types
//	/etc/apache/mime.types
//
// On Windows, MIME types are extracted from the registry.
//
// Text types have the charset parameter set to "utf-8" by default.
func TypeByExtension(ext string) string {
	once.Do(initMime)

	// Case-sensitive lookup.
	if v, ok := mimeTypes.Load(ext); ok {
		return v.(string)
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
			si, _ := mimeTypesLower.Load(strings.ToLower(ext))
			s, _ := si.(string)
			return s
		}
		if 'A' <= c && c <= 'Z' {
			lower = append(lower, c+('a'-'A'))
		} else {
			lower = append(lower, c)
		}
	}
	si, _ := mimeTypesLower.Load(string(lower))
	s, _ := si.(string)
	return s
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
	s, ok := extensions.Load(justType)
	if !ok {
		return nil, nil
	}
	ret := append([]string(nil), s.([]string)...)
	sort.Strings(ret)
	return ret, nil
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

	mimeTypes.Store(extension, mimeType)
	mimeTypesLower.Store(extLower, mimeType)

	extensionsMu.Lock()
	defer extensionsMu.Unlock()
	var exts []string
	if ei, ok := extensions.Load(justType); ok {
		exts = ei.([]string)
	}
	for _, v := range exts {
		if v == extLower {
			return nil
		}
	}
	extensions.Store(justType, append(exts, extLower))
	return nil
}
