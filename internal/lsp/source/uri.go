// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
)

const fileSchemePrefix = "file://"

// URI represents the full uri for a file.
type URI string

// Filename gets the file path for the URI.
// It will return an error if the uri is not valid, or if the URI was not
// a file URI
func (uri URI) Filename() (string, error) {
	s := string(uri)
	if !strings.HasPrefix(s, fileSchemePrefix) {
		return "", fmt.Errorf("only file URI's are supported, got %v", uri)
	}
	s = s[len(fileSchemePrefix):]
	s, err := url.PathUnescape(s)
	if err != nil {
		return s, err
	}
	s = filepath.FromSlash(s)
	return s, nil
}

// ToURI returns a protocol URI for the supplied path.
// It will always have the file scheme.
func ToURI(path string) URI {
	return URI(fileSchemePrefix + filepath.ToSlash(path))
}
