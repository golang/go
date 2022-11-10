// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safefilepath

import (
	"syscall"
	"unicode/utf8"
)

func fromFS(path string) (string, error) {
	if !utf8.ValidString(path) {
		return "", errInvalidPath
	}
	for len(path) > 1 && path[0] == '/' && path[1] == '/' {
		path = path[1:]
	}
	containsSlash := false
	for p := path; p != ""; {
		// Find the next path element.
		i := 0
		dot := -1
		for i < len(p) && p[i] != '/' {
			switch p[i] {
			case 0, '\\', ':':
				return "", errInvalidPath
			case '.':
				if dot < 0 {
					dot = i
				}
			}
			i++
		}
		part := p[:i]
		if i < len(p) {
			containsSlash = true
			p = p[i+1:]
		} else {
			p = ""
		}
		// Trim the extension and look for a reserved name.
		base := part
		if dot >= 0 {
			base = part[:dot]
		}
		if isReservedName(base) {
			if dot < 0 {
				return "", errInvalidPath
			}
			// The path element is a reserved name with an extension.
			// Some Windows versions consider this a reserved name,
			// while others do not. Use FullPath to see if the name is
			// reserved.
			if p, _ := syscall.FullPath(part); len(p) >= 4 && p[:4] == `\\.\` {
				return "", errInvalidPath
			}
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
	if 3 <= len(name) && len(name) <= 4 {
		switch string([]byte{toUpper(name[0]), toUpper(name[1]), toUpper(name[2])}) {
		case "CON", "PRN", "AUX", "NUL":
			return len(name) == 3
		case "COM", "LPT":
			return len(name) == 4 && '1' <= name[3] && name[3] <= '9'
		}
	}
	return false
}

func toUpper(c byte) byte {
	if 'a' <= c && c <= 'z' {
		return c - ('a' - 'A')
	}
	return c
}
