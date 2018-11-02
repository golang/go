// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
)

const fileSchemePrefix = "file://"

// FromURI gets the file path for a given URI.
// It will return an error if the uri is not valid, or if the URI was not
// a file URI
func FromURI(uri protocol.DocumentURI) (string, error) {
	s := string(uri)
	if !strings.HasPrefix(s, fileSchemePrefix) {
		return "", fmt.Errorf("only file URI's are supported, got %v", uri)
	}
	return filepath.FromSlash(s[len(fileSchemePrefix):]), nil
}

// ToURI returns a protocol URI for the supplied path.
// It will always have the file scheme.
func ToURI(path string) protocol.DocumentURI {
	return protocol.DocumentURI(fileSchemePrefix + filepath.ToSlash(path))
}
