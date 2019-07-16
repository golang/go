// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// FS-safe encoding of module paths and versions.
// Copied from cmd/go/internal/module and unexported.

package sumweb

import (
	"fmt"
	"unicode/utf8"
)

// Safe encodings
//
// Module paths appear as substrings of file system paths
// (in the download cache) and of web server URLs in the proxy protocol.
// In general we cannot rely on file systems to be case-sensitive,
// nor can we rely on web servers, since they read from file systems.
// That is, we cannot rely on the file system to keep rsc.io/QUOTE
// and rsc.io/quote separate. Windows and macOS don't.
// Instead, we must never require two different casings of a file path.
// Because we want the download cache to match the proxy protocol,
// and because we want the proxy protocol to be possible to serve
// from a tree of static files (which might be stored on a case-insensitive
// file system), the proxy protocol must never require two different casings
// of a URL path either.
//
// One possibility would be to make the safe encoding be the lowercase
// hexadecimal encoding of the actual path bytes. This would avoid ever
// needing different casings of a file path, but it would be fairly illegible
// to most programmers when those paths appeared in the file system
// (including in file paths in compiler errors and stack traces)
// in web server logs, and so on. Instead, we want a safe encoding that
// leaves most paths unaltered.
//
// The safe encoding is this:
// replace every uppercase letter with an exclamation mark
// followed by the letter's lowercase equivalent.
//
// For example,
// github.com/Azure/azure-sdk-for-go ->  github.com/!azure/azure-sdk-for-go.
// github.com/GoogleCloudPlatform/cloudsql-proxy -> github.com/!google!cloud!platform/cloudsql-proxy
// github.com/Sirupsen/logrus -> github.com/!sirupsen/logrus.
//
// Import paths that avoid upper-case letters are left unchanged.
// Note that because import paths are ASCII-only and avoid various
// problematic punctuation (like : < and >), the safe encoding is also ASCII-only
// and avoids the same problematic punctuation.
//
// Import paths have never allowed exclamation marks, so there is no
// need to define how to encode a literal !.
//
// Although paths are disallowed from using Unicode (see pathOK above),
// the eventual plan is to allow Unicode letters as well, to assume that
// file systems and URLs are Unicode-safe (storing UTF-8), and apply
// the !-for-uppercase convention. Note however that not all runes that
// are different but case-fold equivalent are an upper/lower pair.
// For example, U+004B ('K'), U+006B ('k'), and U+212A ('K' for Kelvin)
// are considered to case-fold to each other. When we do add Unicode
// letters, we must not assume that upper/lower are the only case-equivalent pairs.
// Perhaps the Kelvin symbol would be disallowed entirely, for example.
// Or perhaps it would encode as "!!k", or perhaps as "(212A)".
//
// Also, it would be nice to allow Unicode marks as well as letters,
// but marks include combining marks, and then we must deal not
// only with case folding but also normalization: both U+00E9 ('é')
// and U+0065 U+0301 ('e' followed by combining acute accent)
// look the same on the page and are treated by some file systems
// as the same path. If we do allow Unicode marks in paths, there
// must be some kind of normalization to allow only one canonical
// encoding of any character used in an import path.

// encodePath returns the safe encoding of the given module path.
// It fails if the module path is invalid.
func encodePath(path string) (encoding string, err error) {
	return encodeString(path)
}

// encodeVersion returns the safe encoding of the given module version.
// Versions are allowed to be in non-semver form but must be valid file names
// and not contain exclamation marks.
func encodeVersion(v string) (encoding string, err error) {
	return encodeString(v)
}

func encodeString(s string) (encoding string, err error) {
	haveUpper := false
	for _, r := range s {
		if r == '!' || r >= utf8.RuneSelf {
			// This should be disallowed by CheckPath, but diagnose anyway.
			// The correctness of the encoding loop below depends on it.
			return "", fmt.Errorf("internal error: inconsistency in EncodePath")
		}
		if 'A' <= r && r <= 'Z' {
			haveUpper = true
		}
	}

	if !haveUpper {
		return s, nil
	}

	var buf []byte
	for _, r := range s {
		if 'A' <= r && r <= 'Z' {
			buf = append(buf, '!', byte(r+'a'-'A'))
		} else {
			buf = append(buf, byte(r))
		}
	}
	return string(buf), nil
}

// decodePath returns the module path of the given safe encoding.
// It fails if the encoding is invalid or encodes an invalid path.
func decodePath(encoding string) (path string, err error) {
	path, ok := decodeString(encoding)
	if !ok {
		return "", fmt.Errorf("invalid module path encoding %q", encoding)
	}
	return path, nil
}

// decodeVersion returns the version string for the given safe encoding.
// It fails if the encoding is invalid or encodes an invalid version.
// Versions are allowed to be in non-semver form but must be valid file names
// and not contain exclamation marks.
func decodeVersion(encoding string) (v string, err error) {
	v, ok := decodeString(encoding)
	if !ok {
		return "", fmt.Errorf("invalid version encoding %q", encoding)
	}
	return v, nil
}

func decodeString(encoding string) (string, bool) {
	var buf []byte

	bang := false
	for _, r := range encoding {
		if r >= utf8.RuneSelf {
			return "", false
		}
		if bang {
			bang = false
			if r < 'a' || 'z' < r {
				return "", false
			}
			buf = append(buf, byte(r+'A'-'a'))
			continue
		}
		if r == '!' {
			bang = true
			continue
		}
		if 'A' <= r && r <= 'Z' {
			return "", false
		}
		buf = append(buf, byte(r))
	}
	if bang {
		return "", false
	}
	return string(buf), true
}
