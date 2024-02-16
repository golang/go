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
		for i < len(p) && p[i] != '/' {
			switch p[i] {
			case 0, '\\', ':':
				return "", errInvalidPath
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
		if IsReservedName(part) {
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

// IsReservedName reports if name is a Windows reserved device name.
// It does not detect names with an extension, which are also reserved on some Windows versions.
//
// For details, search for PRN in
// https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file.
func IsReservedName(name string) bool {
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
	// Some Windows versions consider this a reserved name,
	// while others do not. Use FullPath to see if the name is
	// reserved.
	if p, _ := syscall.FullPath(name); len(p) >= 4 && p[:4] == `\\.\` {
		return true
	}
	return false
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
