// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepathlite

import (
	"internal/bytealg"
	"internal/stringslite"
	"internal/syscall/windows"
	"syscall"
)

const (
	Separator     = '\\' // OS-specific path separator
	ListSeparator = ';'  // OS-specific path list separator
)

func IsPathSeparator(c uint8) bool {
	return c == '\\' || c == '/'
}

func isLocal(path string) bool {
	if path == "" {
		return false
	}
	if IsPathSeparator(path[0]) {
		// Path rooted in the current drive.
		return false
	}
	if stringslite.IndexByte(path, ':') >= 0 {
		// Colons are only valid when marking a drive letter ("C:foo").
		// Rejecting any path with a colon is conservative but safe.
		return false
	}
	hasDots := false // contains . or .. path elements
	for p := path; p != ""; {
		var part string
		part, p, _ = cutPath(p)
		if part == "." || part == ".." {
			hasDots = true
		}
		if isReservedName(part) {
			return false
		}
	}
	if hasDots {
		path = Clean(path)
	}
	if path == ".." || stringslite.HasPrefix(path, `..\`) {
		return false
	}
	return true
}

func localize(path string) (string, error) {
	for i := 0; i < len(path); i++ {
		switch path[i] {
		case ':', '\\', 0:
			return "", errInvalidPath
		}
	}
	containsSlash := false
	for p := path; p != ""; {
		// Find the next path element.
		var element string
		i := bytealg.IndexByteString(p, '/')
		if i < 0 {
			element = p
			p = ""
		} else {
			containsSlash = true
			element = p[:i]
			p = p[i+1:]
		}
		if isReservedName(element) {
			return "", errInvalidPath
		}
	}
	if containsSlash {
		// We can't depend on strings, so substitute \ for / manually.
		buf := []byte(path)
		for i, b := range buf {
			if b == '/' {
				buf[i] = '\\'
			}
		}
		path = string(buf)
	}
	return path, nil
}

// isReservedName reports if name is a Windows reserved device name.
// It does not detect names with an extension, which are also reserved on some Windows versions.
//
// For details, search for PRN in
// https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file.
func isReservedName(name string) bool {
	// Device names can have arbitrary trailing characters following a dot or colon.
	base := name
	for i := 0; i < len(base); i++ {
		switch base[i] {
		case ':', '.':
			base = base[:i]
		}
	}
	// Trailing spaces in the last path element are ignored.
	for len(base) > 0 && base[len(base)-1] == ' ' {
		base = base[:len(base)-1]
	}
	if !isReservedBaseName(base) {
		return false
	}
	if len(base) == len(name) {
		return true
	}
	// The path element is a reserved name with an extension.
	// Since Windows 11, reserved names with extensions are no
	// longer reserved. For example, "CON.txt" is a valid file
	// name. Use RtlIsDosDeviceName_U to see if the name is reserved.
	p, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		return false
	}
	return windows.RtlIsDosDeviceName_U(p) > 0
}

func isReservedBaseName(name string) bool {
	if len(name) == 3 {
		switch string([]byte{toUpper(name[0]), toUpper(name[1]), toUpper(name[2])}) {
		case "CON", "PRN", "AUX", "NUL":
			return true
		}
	}
	if len(name) >= 4 {
		switch string([]byte{toUpper(name[0]), toUpper(name[1]), toUpper(name[2])}) {
		case "COM", "LPT":
			if len(name) == 4 && '1' <= name[3] && name[3] <= '9' {
				return true
			}
			// Superscript ¹, ², and ³ are considered numbers as well.
			switch name[3:] {
			case "\u00b2", "\u00b3", "\u00b9":
				return true
			}
			return false
		}
	}

	// Passing CONIN$ or CONOUT$ to CreateFile opens a console handle.
	// https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea#consoles
	//
	// While CONIN$ and CONOUT$ aren't documented as being files,
	// they behave the same as CON. For example, ./CONIN$ also opens the console input.
	if len(name) == 6 && name[5] == '$' && equalFold(name, "CONIN$") {
		return true
	}
	if len(name) == 7 && name[6] == '$' && equalFold(name, "CONOUT$") {
		return true
	}
	return false
}

func equalFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if toUpper(a[i]) != toUpper(b[i]) {
			return false
		}
	}
	return true
}

func toUpper(c byte) byte {
	if 'a' <= c && c <= 'z' {
		return c - ('a' - 'A')
	}
	return c
}

// IsAbs reports whether the path is absolute.
func IsAbs(path string) (b bool) {
	l := volumeNameLen(path)
	if l == 0 {
		return false
	}
	// If the volume name starts with a double slash, this is an absolute path.
	if IsPathSeparator(path[0]) && IsPathSeparator(path[1]) {
		return true
	}
	path = path[l:]
	if path == "" {
		return false
	}
	return IsPathSeparator(path[0])
}

// volumeNameLen returns length of the leading volume name on Windows.
// It returns 0 elsewhere.
//
// See:
// https://learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats
// https://googleprojectzero.blogspot.com/2016/02/the-definitive-guide-on-win32-to-nt.html
func volumeNameLen(path string) int {
	switch {
	case len(path) >= 2 && path[1] == ':':
		// Path starts with a drive letter.
		//
		// Not all Windows functions necessarily enforce the requirement that
		// drive letters be in the set A-Z, and we don't try to here.
		//
		// We don't handle the case of a path starting with a non-ASCII character,
		// in which case the "drive letter" might be multiple bytes long.
		return 2

	case len(path) == 0 || !IsPathSeparator(path[0]):
		// Path does not have a volume component.
		return 0

	case pathHasPrefixFold(path, `\\.\UNC`):
		// We're going to treat the UNC host and share as part of the volume
		// prefix for historical reasons, but this isn't really principled;
		// Windows's own GetFullPathName will happily remove the first
		// component of the path in this space, converting
		// \\.\unc\a\b\..\c into \\.\unc\a\c.
		return uncLen(path, len(`\\.\UNC\`))

	case pathHasPrefixFold(path, `\\.`) ||
		pathHasPrefixFold(path, `\\?`) || pathHasPrefixFold(path, `\??`):
		// Path starts with \\.\, and is a Local Device path; or
		// path starts with \\?\ or \??\ and is a Root Local Device path.
		//
		// We treat the next component after the \\.\ prefix as
		// part of the volume name, which means Clean(`\\?\c:\`)
		// won't remove the trailing \. (See #64028.)
		if len(path) == 3 {
			return 3 // exactly \\.
		}
		_, rest, ok := cutPath(path[4:])
		if !ok {
			return len(path)
		}
		return len(path) - len(rest) - 1

	case len(path) >= 2 && IsPathSeparator(path[1]):
		// Path starts with \\, and is a UNC path.
		return uncLen(path, 2)
	}
	return 0
}

// pathHasPrefixFold tests whether the path s begins with prefix,
// ignoring case and treating all path separators as equivalent.
// If s is longer than prefix, then s[len(prefix)] must be a path separator.
func pathHasPrefixFold(s, prefix string) bool {
	if len(s) < len(prefix) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		if IsPathSeparator(prefix[i]) {
			if !IsPathSeparator(s[i]) {
				return false
			}
		} else if toUpper(prefix[i]) != toUpper(s[i]) {
			return false
		}
	}
	if len(s) > len(prefix) && !IsPathSeparator(s[len(prefix)]) {
		return false
	}
	return true
}

// uncLen returns the length of the volume prefix of a UNC path.
// prefixLen is the prefix prior to the start of the UNC host;
// for example, for "//host/share", the prefixLen is len("//")==2.
func uncLen(path string, prefixLen int) int {
	count := 0
	for i := prefixLen; i < len(path); i++ {
		if IsPathSeparator(path[i]) {
			count++
			if count == 2 {
				return i
			}
		}
	}
	return len(path)
}

// cutPath slices path around the first path separator.
func cutPath(path string) (before, after string, found bool) {
	for i := range path {
		if IsPathSeparator(path[i]) {
			return path[:i], path[i+1:], true
		}
	}
	return path, "", false
}

// postClean adjusts the results of Clean to avoid turning a relative path
// into an absolute or rooted one.
func postClean(out *lazybuf) {
	if out.volLen != 0 || out.buf == nil {
		return
	}
	// If a ':' appears in the path element at the start of a path,
	// insert a .\ at the beginning to avoid converting relative paths
	// like a/../c: into c:.
	for _, c := range out.buf {
		if IsPathSeparator(c) {
			break
		}
		if c == ':' {
			out.prepend('.', Separator)
			return
		}
	}
	// If a path begins with \??\, insert a \. at the beginning
	// to avoid converting paths like \a\..\??\c:\x into \??\c:\x
	// (equivalent to c:\x).
	if len(out.buf) >= 3 && IsPathSeparator(out.buf[0]) && out.buf[1] == '?' && out.buf[2] == '?' {
		out.prepend(Separator, '.')
	}
}
