// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package filepath

import (
	"strings"
)

// HasPrefix exists for historical compatibility and should not be used.
//
// Deprecated: HasPrefix does not respect path boundaries and
// does not ignore case when required.
func HasPrefix(p, prefix string) bool {
	return strings.HasPrefix(p, prefix)
}

func splitList(path string) []string {
	if path == "" {
		return []string{}
	}
	return strings.Split(path, string(ListSeparator))
}

func abs(path string) (string, error) {
	return unixAbs(path)
}

func join(elem []string) string {
	// If there's a bug here, fix the logic in ./path_plan9.go too.
	for i, e := range elem {
		if e != "" {
			return Clean(strings.Join(elem[i:], string(Separator)))
		}
	}
	return ""
}

func sameWord(a, b string) bool {
	return a == b
}
