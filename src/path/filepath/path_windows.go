// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath

import (
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

// isReservedName reports if name is a Windows reserved device name or a console handle.
// It does not detect names with an extension, which are also reserved on some Windows versions.
//
// For details, search for PRN in
// https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file.
func isReservedName(name string) bool {
	if 3 <= len(name) && len(name) <= 4 {
		switch string([]byte{toUpper(name[0]), toUpper(name[1]), toUpper(name[2])}) {
		case "CON", "PRN", "AUX", "NUL":
			return len(name) == 3
		case "COM", "LPT":
			return len(name) == 4 && '1' <= name[3] && name[3] <= '9'
		}
	}
	// Passing CONIN$ or CONOUT$ to CreateFile opens a console handle.
	// https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea#consoles
	//
	// While CONIN$ and CONOUT$ aren't documented as being files,
	// they behave the same as CON. For example, ./CONIN$ also opens the console input.
	if len(name) == 6 && name[5] == '$' && strings.EqualFold(name, "CONIN$") {
		return true
	}
	if len(name) == 7 && name[6] == '$' && strings.EqualFold(name, "CONOUT$") {
		return true
	}
	return false
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
		// Trim the extension and look for a reserved name.
		base, _, hasExt := strings.Cut(part, ".")
		if isReservedName(base) {
			if !hasExt {
				return false
			}
			// The path element is a reserved name with an extension. Some Windows
			// versions consider this a reserved name, while others do not. Use
			// FullPath to see if the name is reserved.
			//
			// FullPath will convert references to reserved device names to their
			// canonical form: \\.\${DEVICE_NAME}
			//
			// FullPath does not perform this conversion for paths which contain
			// a reserved device name anywhere other than in the last element,
			// so check the part rather than the full path.
			if p, _ := syscall.FullPath(part); len(p) >= 4 && p[:4] == `\\.\` {
				return false
			}
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
// See: https://learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats
func volumeNameLen(path string) int {
	if len(path) < 2 {
		return 0
	}
	// with drive letter
	c := path[0]
	if path[1] == ':' && ('a' <= c && c <= 'z' || 'A' <= c && c <= 'Z') {
		return 2
	}
	// UNC and DOS device paths start with two slashes.
	if !isSlash(path[0]) || !isSlash(path[1]) {
		return 0
	}
	rest := path[2:]
	p1, rest, _ := cutPath(rest)
	p2, rest, ok := cutPath(rest)
	if !ok {
		return len(path)
	}
	if p1 != "." && p1 != "?" {
		// This is a UNC path: \\${HOST}\${SHARE}\
		return len(path) - len(rest) - 1
	}
	// This is a DOS device path.
	if len(p2) == 3 && toUpper(p2[0]) == 'U' && toUpper(p2[1]) == 'N' && toUpper(p2[2]) == 'C' {
		// This is a DOS device path that links to a UNC: \\.\UNC\${HOST}\${SHARE}\
		_, rest, _ = cutPath(rest)  // host
		_, rest, ok = cutPath(rest) // share
		if !ok {
			return len(path)
		}
	}
	return len(path) - len(rest) - 1
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
