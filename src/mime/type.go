// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mime implements parts of the MIME spec.
package mime

import (
	"fmt"
	"slices"
	"strings"
	"sync"
)

var (
	mimeTypes      sync.Map // map[string]string; ".Z" => "application/x-compress"
	mimeTypesLower sync.Map // map[string]string; ".z" => "application/x-compress"

	// extensions maps from MIME type to list of lowercase file
	// extensions: "image/jpeg" => [".jfif", ".jpg", ".jpeg", ".pjp", ".pjpeg"]
	extensionsMu sync.Mutex // Guards stores (but not loads) on extensions.
	extensions   sync.Map   // map[string][]string; slice values are append-only.
)

// setMimeTypes is used by initMime's non-test path, and by tests.
func setMimeTypes(lowerExt, mixExt map[string]string) {
	mimeTypes.Clear()
	mimeTypesLower.Clear()
	extensions.Clear()

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

// A type is listed here if both Firefox and Chrome included them in their own
// lists.  In the case where they contradict they are deconflicted using IANA's
// listed media types https://www.iana.org/assignments/media-types/media-types.xhtml
//
// Chrome's MIME mappings to file extensions are defined at
// https://chromium.googlesource.com/chromium/src.git/+/refs/heads/main/net/base/mime_util.cc
//
// Firefox's MIME types can be found at
// https://github.com/mozilla-firefox/firefox/blob/main/netwerk/mime/nsMimeTypes.h
// and the mappings to file extensions at
// https://github.com/mozilla-firefox/firefox/blob/main/uriloader/exthandler/nsExternalHelperAppService.cpp
var builtinTypesLower = map[string]string{
	".ai":    "application/postscript",
	".apk":   "application/vnd.android.package-archive",
	".apng":  "image/apng",
	".avif":  "image/avif",
	".bin":   "application/octet-stream",
	".bmp":   "image/bmp",
	".com":   "application/octet-stream",
	".css":   "text/css; charset=utf-8",
	".csv":   "text/csv; charset=utf-8",
	".doc":   "application/msword",
	".docx":  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	".ehtml": "text/html; charset=utf-8",
	".eml":   "message/rfc822",
	".eps":   "application/postscript",
	".exe":   "application/octet-stream",
	".flac":  "audio/flac",
	".gif":   "image/gif",
	".gz":    "application/gzip",
	".htm":   "text/html; charset=utf-8",
	".html":  "text/html; charset=utf-8",
	".ico":   "image/vnd.microsoft.icon",
	".ics":   "text/calendar; charset=utf-8",
	".jfif":  "image/jpeg",
	".jpeg":  "image/jpeg",
	".jpg":   "image/jpeg",
	".js":    "text/javascript; charset=utf-8",
	".json":  "application/json",
	".m4a":   "audio/mp4",
	".mjs":   "text/javascript; charset=utf-8",
	".mp3":   "audio/mpeg",
	".mp4":   "video/mp4",
	".oga":   "audio/ogg",
	".ogg":   "audio/ogg",
	".ogv":   "video/ogg",
	".opus":  "audio/ogg",
	".pdf":   "application/pdf",
	".pjp":   "image/jpeg",
	".pjpeg": "image/jpeg",
	".png":   "image/png",
	".ppt":   "application/vnd.ms-powerpoint",
	".pptx":  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
	".ps":    "application/postscript",
	".rdf":   "application/rdf+xml",
	".rtf":   "application/rtf",
	".shtml": "text/html; charset=utf-8",
	".svg":   "image/svg+xml",
	".text":  "text/plain; charset=utf-8",
	".tif":   "image/tiff",
	".tiff":  "image/tiff",
	".txt":   "text/plain; charset=utf-8",
	".vtt":   "text/vtt; charset=utf-8",
	".wasm":  "application/wasm",
	".wav":   "audio/wav",
	".webm":  "audio/webm",
	".webp":  "image/webp",
	".xbl":   "text/xml; charset=utf-8",
	".xbm":   "image/x-xbitmap",
	".xht":   "application/xhtml+xml",
	".xhtml": "application/xhtml+xml",
	".xls":   "application/vnd.ms-excel",
	".xlsx":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
	".xml":   "text/xml; charset=utf-8",
	".xsl":   "text/xml; charset=utf-8",
	".zip":   "application/zip",
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
//	/etc/httpd/conf/mime.types
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
//	/etc/httpd/conf/mime.types
//
// On Windows, extensions are extracted from the registry.
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
	slices.Sort(ret)
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
