// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"strings"
)

// tagOptions is the string following a comma in a struct field's "json"
// tag, or the empty string. It does not include the leading comma.
type tagOptions string

// parseTag splits a struct field's json tag into its name and
// comma-separated options.
func parseTag(tag string) (string, tagOptions) {
	tag, opt := cutTag(tag)
	return tag, tagOptions(opt)
}

func cutTag(tag string) (string, string) {
	if i := strings.IndexByte(tag, ','); i >= 0 {
		return tag[:i], tag[i+1:]
	}
	return tag, ""
}

// Contains reports whether a comma-separated list of options
// contains a particular substr flag. substr must be surrounded by a
// string boundary or commas.
func (o tagOptions) Contains(optionName string) bool {
	if len(o) == 0 {
		return false
	}
	s := string(o)
	for s != "" {
		var name string
		name, s = cutTag(s)
		if name == optionName {
			return true
		}
	}
	return false
}
