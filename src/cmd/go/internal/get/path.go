// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package get

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// The following functions are copied verbatim from cmd/go/internal/module/module.go,
// with a change to additionally reject Windows short-names,
// and one to accept arbitrary letters (golang.org/issue/29101).
//
// TODO(bcmills): After the call site for this function is backported,
// consolidate this back down to a single copy.
//
// NOTE: DO NOT MERGE THESE UNTIL WE DECIDE ABOUT ARBITRARY LETTERS IN MODULE MODE.

// CheckImportPath checks that an import path is valid.
func CheckImportPath(path string) error {
	if err := checkPath(path, false); err != nil {
		return fmt.Errorf("malformed import path %q: %v", path, err)
	}
	return nil
}

// checkPath checks that a general path is valid.
// It returns an error describing why but not mentioning path.
// Because these checks apply to both module paths and import paths,
// the caller is expected to add the "malformed ___ path %q: " prefix.
// fileName indicates whether the final element of the path is a file name
// (as opposed to a directory name).
func checkPath(path string, fileName bool) error {
	if !utf8.ValidString(path) {
		return fmt.Errorf("invalid UTF-8")
	}
	if path == "" {
		return fmt.Errorf("empty string")
	}
	if path[0] == '-' {
		return fmt.Errorf("leading dash")
	}
	if strings.Contains(path, "..") {
		return fmt.Errorf("double dot")
	}
	if strings.Contains(path, "//") {
		return fmt.Errorf("double slash")
	}
	if path[len(path)-1] == '/' {
		return fmt.Errorf("trailing slash")
	}
	elemStart := 0
	for i, r := range path {
		if r == '/' {
			if err := checkElem(path[elemStart:i], fileName); err != nil {
				return err
			}
			elemStart = i + 1
		}
	}
	if err := checkElem(path[elemStart:], fileName); err != nil {
		return err
	}
	return nil
}

// checkElem checks whether an individual path element is valid.
// fileName indicates whether the element is a file name (not a directory name).
func checkElem(elem string, fileName bool) error {
	if elem == "" {
		return fmt.Errorf("empty path element")
	}
	if strings.Count(elem, ".") == len(elem) {
		return fmt.Errorf("invalid path element %q", elem)
	}
	if elem[0] == '.' && !fileName {
		return fmt.Errorf("leading dot in path element")
	}
	if elem[len(elem)-1] == '.' {
		return fmt.Errorf("trailing dot in path element")
	}

	charOK := pathOK
	if fileName {
		charOK = fileNameOK
	}
	for _, r := range elem {
		if !charOK(r) {
			return fmt.Errorf("invalid char %q", r)
		}
	}

	// Windows disallows a bunch of path elements, sadly.
	// See https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file
	short := elem
	if i := strings.Index(short, "."); i >= 0 {
		short = short[:i]
	}
	for _, bad := range badWindowsNames {
		if strings.EqualFold(bad, short) {
			return fmt.Errorf("disallowed path element %q", elem)
		}
	}

	// Reject path components that look like Windows short-names.
	// Those usually end in a tilde followed by one or more ASCII digits.
	if tilde := strings.LastIndexByte(short, '~'); tilde >= 0 && tilde < len(short)-1 {
		suffix := short[tilde+1:]
		suffixIsDigits := true
		for _, r := range suffix {
			if r < '0' || r > '9' {
				suffixIsDigits = false
				break
			}
		}
		if suffixIsDigits {
			return fmt.Errorf("trailing tilde and digits in path element")
		}
	}

	return nil
}

// pathOK reports whether r can appear in an import path element.
//
// NOTE: This function DIVERGES from module mode pathOK by accepting Unicode letters.
func pathOK(r rune) bool {
	if r < utf8.RuneSelf {
		return r == '+' || r == '-' || r == '.' || r == '_' || r == '~' ||
			'0' <= r && r <= '9' ||
			'A' <= r && r <= 'Z' ||
			'a' <= r && r <= 'z'
	}
	return unicode.IsLetter(r)
}

// fileNameOK reports whether r can appear in a file name.
// For now we allow all Unicode letters but otherwise limit to pathOK plus a few more punctuation characters.
// If we expand the set of allowed characters here, we have to
// work harder at detecting potential case-folding and normalization collisions.
// See note about "safe encoding" below.
func fileNameOK(r rune) bool {
	if r < utf8.RuneSelf {
		// Entire set of ASCII punctuation, from which we remove characters:
		//     ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
		// We disallow some shell special characters: " ' * < > ? ` |
		// (Note that some of those are disallowed by the Windows file system as well.)
		// We also disallow path separators / : and \ (fileNameOK is only called on path element characters).
		// We allow spaces (U+0020) in file names.
		const allowed = "!#$%&()+,-.=@[]^_{}~ "
		if '0' <= r && r <= '9' || 'A' <= r && r <= 'Z' || 'a' <= r && r <= 'z' {
			return true
		}
		for i := 0; i < len(allowed); i++ {
			if rune(allowed[i]) == r {
				return true
			}
		}
		return false
	}
	// It may be OK to add more ASCII punctuation here, but only carefully.
	// For example Windows disallows < > \, and macOS disallows :, so we must not allow those.
	return unicode.IsLetter(r)
}

// badWindowsNames are the reserved file path elements on Windows.
// See https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file
var badWindowsNames = []string{
	"CON",
	"PRN",
	"AUX",
	"NUL",
	"COM1",
	"COM2",
	"COM3",
	"COM4",
	"COM5",
	"COM6",
	"COM7",
	"COM8",
	"COM9",
	"LPT1",
	"LPT2",
	"LPT3",
	"LPT4",
	"LPT5",
	"LPT6",
	"LPT7",
	"LPT8",
	"LPT9",
}
