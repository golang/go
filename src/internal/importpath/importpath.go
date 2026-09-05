// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importpath

import (
	"strings"
	"unicode"
)

const illegalChars = `!"#$%&'()*,:;<=>?[\]^{|}` + "`\uFFFD"

func isValidImportPathChar(r rune) bool {
	return unicode.IsGraphic(r) && !unicode.IsSpace(r) && !strings.ContainsRune(illegalChars, r)
}

// IsValidImport checks if the import is a valid import using the more strict
// checks allowed by the implementation restriction in https://go.dev/ref/spec#Import_declarations.
// This logic was previously duplicated across several packages.
func IsValidImport(s string) bool {
	for _, r := range s {
		if !isValidImportPathChar(r) {
			return false
		}
	}
	return s != ""
}
