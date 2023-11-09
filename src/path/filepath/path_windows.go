// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import (
	"internal/safefilepath"
	"os"
	"strings"
	"syscall"
)

func isSlash(c uint8) bool {
	return c == '\\' || c == '/'
}

func toUpper(c byte) byte {
	if 'a' <= c && c <= 'z' {
		return c - ('a' - 'A')
	}
	return c
}

func isLocal(path string) bool {
	if path == "" {
		return false
	}
	if isSlash(path[0]) {
		// Path rooted in the current drive.
		return false
	}
	if strings.IndexByte(path, ':') >= 0 {
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
		if safefilepath.IsReservedName(part) {
			return false
		}
	}
	if hasDots {
		path = Clean(path)
	}
	if path == ".." || strings.HasPrefix(path, `..\`) {
		return false
	}
	return true
}

// IsAbs reports whether the path is absolute.
func IsAbs(path string) (b bool) {
	l := volumeNameLen(path)
	if l == 0 {
		return false
	}
	// If the volume name starts with a double slash, this is an absolute path.
	if isSlash(path[0]) && isSlash(path[1]) {
		return true
	}
	path = path[l:]
	if path == "" {
		return false
	}
	return isSlash(path[0])
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

	case len(path) == 0 || !isSlash(path[0]):
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

	case len(path) >= 2 && isSlash(path[1]):
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
		if isSlash(prefix[i]) {
			if !isSlash(s[i]) {
				return false
			}
		} else if toUpper(prefix[i]) != toUpper(s[i]) {
			return false
		}
	}
	if len(s) > len(prefix) && !isSlash(s[len(prefix)]) {
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
		if isSlash(path[i]) {
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
		if isSlash(path[i]) {
			return path[:i], path[i+1:], true
		}
	}
	return path, "", false
}

// HasPrefix exists for historical compatibility and should not be used.
//
// Deprecated: HasPrefix does not respect path boundaries and
// does not ignore case when required.
func HasPrefix(p, prefix string) bool {
	if strings.HasPrefix(p, prefix) {
		return true
	}
	return strings.HasPrefix(strings.ToLower(p), strings.ToLower(prefix))
}

func splitList(path string) []string {
	// The same implementation is used in LookPath in os/exec;
	// consider changing os/exec when changing this.

	if path == "" {
		return []string{}
	}

	// Split path, respecting but preserving quotes.
	list := []string{}
	start := 0
	quo := false
	for i := 0; i < len(path); i++ {
		switch c := path[i]; {
		case c == '"':
			quo = !quo
		case c == ListSeparator && !quo:
			list = append(list, path[start:i])
			start = i + 1
		}
	}
	list = append(list, path[start:])

	// Remove quotes.
	for i, s := range list {
		list[i] = strings.ReplaceAll(s, `"`, ``)
	}

	return list
}

func abs(path string) (string, error) {
	if path == "" {
		// syscall.FullPath returns an error on empty path, because it's not a valid path.
		// To implement Abs behavior of returning working directory on empty string input,
		// special-case empty path by changing it to "." path. See golang.org/issue/24441.
		path = "."
	}
	fullPath, err := syscall.FullPath(path)
	if err != nil {
		return "", err
	}
	return Clean(fullPath), nil
}

func join(elem []string) string {
	var b strings.Builder
	var lastChar byte
	for _, e := range elem {
		switch {
		case b.Len() == 0:
			// Add the first non-empty path element unchanged.
		case isSlash(lastChar):
			// If the path ends in a slash, strip any leading slashes from the next
			// path element to avoid creating a UNC path (any path starting with "\\")
			// from non-UNC elements.
			//
			// The correct behavior for Join when the first element is an incomplete UNC
			// path (for example, "\\") is underspecified. We currently join subsequent
			// elements so Join("\\", "host", "share") produces "\\host\share".
			for len(e) > 0 && isSlash(e[0]) {
				e = e[1:]
			}
			// If the path is \ and the next path element is ??,
			// add an extra .\ to create \.\?? rather than \??\
			// (a Root Local Device path).
			if b.Len() == 1 && pathHasPrefixFold(e, "??") {
				b.WriteString(`.\`)
			}
		case lastChar == ':':
			// If the path ends in a colon, keep the path relative to the current directory
			// on a drive and don't add a separator. Preserve leading slashes in the next
			// path element, which may make the path absolute.
			//
			// 	Join(`C:`, `f`) = `C:f`
			//	Join(`C:`, `\f`) = `C:\f`
		default:
			// In all other cases, add a separator between elements.
			b.WriteByte('\\')
			lastChar = '\\'
		}
		if len(e) > 0 {
			b.WriteString(e)
			lastChar = e[len(e)-1]
		}
	}
	if b.Len() == 0 {
		return ""
	}
	return Clean(b.String())
}

// joinNonEmpty is like join, but it assumes that the first element is non-empty.
func joinNonEmpty(elem []string) string {
	if len(elem[0]) == 2 && elem[0][1] == ':' {
		// First element is drive letter without terminating slash.
		// Keep path relative to current directory on that drive.
		// Skip empty elements.
		i := 1
		for ; i < len(elem); i++ {
			if elem[i] != "" {
				break
			}
		}
		return Clean(elem[0] + strings.Join(elem[i:], string(Separator)))
	}
	// The following logic prevents Join from inadvertently creating a
	// UNC path on Windows. Unless the first element is a UNC path, Join
	// shouldn't create a UNC path. See golang.org/issue/9167.
	p := Clean(strings.Join(elem, string(Separator)))
	if !isUNC(p) {
		return p
	}
	// p == UNC only allowed when the first element is a UNC path.
	head := Clean(elem[0])
	if isUNC(head) {
		return p
	}
	// head + tail == UNC, but joining two non-UNC paths should not result
	// in a UNC path. Undo creation of UNC path.
	tail := Clean(strings.Join(elem[1:], string(Separator)))
	if head[len(head)-1] == Separator {
		return head + tail
	}
	return head + string(Separator) + tail
}

// isUNC reports whether path is a UNC path.
func isUNC(path string) bool {
	return len(path) > 1 && isSlash(path[0]) && isSlash(path[1])
}

func sameWord(a, b string) bool {
	return strings.EqualFold(a, b)
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
		if os.IsPathSeparator(c) {
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
	if len(out.buf) >= 3 && os.IsPathSeparator(out.buf[0]) && out.buf[1] == '?' && out.buf[2] == '?' {
		out.prepend(Separator, '.')
	}
}
