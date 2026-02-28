// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package str

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// HasPathPrefix reports whether the slash-separated path s
// begins with the elements in prefix.
func HasPathPrefix(s, prefix string) bool {
	if len(s) == len(prefix) {
		return s == prefix
	}
	if prefix == "" {
		return true
	}
	if len(s) > len(prefix) {
		if prefix[len(prefix)-1] == '/' || s[len(prefix)] == '/' {
			return s[:len(prefix)] == prefix
		}
	}
	return false
}

// HasFilePathPrefix reports whether the filesystem path s
// begins with the elements in prefix.
//
// HasFilePathPrefix is case-sensitive (except for volume names) even if the
// filesystem is not, does not apply Unicode normalization even if the
// filesystem does, and assumes that all path separators are canonicalized to
// filepath.Separator (as returned by filepath.Clean).
func HasFilePathPrefix(s, prefix string) bool {
	sv := filepath.VolumeName(s)
	pv := filepath.VolumeName(prefix)

	// Strip the volume from both paths before canonicalizing sv and pv:
	// it's unlikely that strings.ToUpper will change the length of the string,
	// but doesn't seem impossible.
	s = s[len(sv):]
	prefix = prefix[len(pv):]

	// Always treat Windows volume names as case-insensitive, even though
	// we don't treat the rest of the path as such.
	//
	// TODO(bcmills): Why do we care about case only for the volume name? It's
	// been this way since https://go.dev/cl/11316, but I don't understand why
	// that problem doesn't apply to case differences in the entire path.
	if sv != pv {
		sv = strings.ToUpper(sv)
		pv = strings.ToUpper(pv)
	}

	switch {
	default:
		return false
	case sv != pv:
		return false
	case len(s) == len(prefix):
		return s == prefix
	case prefix == "":
		return true
	case len(s) > len(prefix):
		if prefix[len(prefix)-1] == filepath.Separator {
			return strings.HasPrefix(s, prefix)
		}
		return s[len(prefix)] == filepath.Separator && s[:len(prefix)] == prefix
	}
}

// TrimFilePathPrefix returns s without the leading path elements in prefix,
// such that joining the string to prefix produces s.
//
// If s does not start with prefix (HasFilePathPrefix with the same arguments
// returns false), TrimFilePathPrefix returns s. If s equals prefix,
// TrimFilePathPrefix returns "".
func TrimFilePathPrefix(s, prefix string) string {
	if prefix == "" {
		// Trimming the empty string from a path should join to produce that path.
		// (Trim("/tmp/foo", "") should give "/tmp/foo", not "tmp/foo".)
		return s
	}
	if !HasFilePathPrefix(s, prefix) {
		return s
	}

	trimmed := s[len(prefix):]
	if len(trimmed) > 0 && os.IsPathSeparator(trimmed[0]) {
		if runtime.GOOS == "windows" && prefix == filepath.VolumeName(prefix) && len(prefix) == 2 && prefix[1] == ':' {
			// Joining a relative path to a bare Windows drive letter produces a path
			// relative to the working directory on that drive, but the original path
			// was absolute, not relative. Keep the leading path separator so that it
			// remains absolute when joined to prefix.
		} else {
			// Prefix ends in a regular path element, so strip the path separator that
			// follows it.
			trimmed = trimmed[1:]
		}
	}
	return trimmed
}

// WithFilePathSeparator returns s with a trailing path separator, or the empty
// string if s is empty.
func WithFilePathSeparator(s string) string {
	if s == "" || os.IsPathSeparator(s[len(s)-1]) {
		return s
	}
	return s + string(filepath.Separator)
}

// QuoteGlob returns s with all Glob metacharacters quoted.
// We don't try to handle backslash here, as that can appear in a
// file path on Windows.
func QuoteGlob(s string) string {
	if !strings.ContainsAny(s, `*?[]`) {
		return s
	}
	var sb strings.Builder
	for _, c := range s {
		switch c {
		case '*', '?', '[', ']':
			sb.WriteByte('\\')
		}
		sb.WriteRune(c)
	}
	return sb.String()
}
