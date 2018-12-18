// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"net/url"
	"path/filepath"
	"runtime"
	"strings"
	"unicode"
)

const fileScheme = "file"

// URI represents the full URI for a file.
type URI string

// Filename gets the file path for the URI.
// It will return an error if the uri is not valid, or if the URI was not
// a file URI
func (uri URI) Filename() (string, error) {
	filename, err := filename(uri)
	if err != nil {
		return "", err
	}
	return filepath.FromSlash(filename), nil
}

func filename(uri URI) (string, error) {
	u, err := url.ParseRequestURI(string(uri))
	if err != nil {
		return "", err
	}
	if u.Scheme != fileScheme {
		return "", fmt.Errorf("only file URIs are supported, got %v", u.Scheme)
	}
	if isWindowsDriveURI(u.Path) {
		u.Path = u.Path[1:]
	}
	return u.Path, nil
}

// ToURI returns a protocol URI for the supplied path.
// It will always have the file scheme.
func ToURI(path string) URI {
	u := toURI(path)
	u.Path = filepath.ToSlash(u.Path)
	return URI(u.String())
}

func toURI(path string) *url.URL {
	// Handle standard library paths that contain the literal "$GOROOT".
	// TODO(rstambler): The go/packages API should allow one to determine a user's $GOROOT.
	const prefix = "$GOROOT"
	if len(path) >= len(prefix) && strings.EqualFold(prefix, path[:len(prefix)]) {
		suffix := path[len(prefix):]
		path = runtime.GOROOT() + suffix
	}
	if isWindowsDrivePath(path) {
		path = "/" + path
	}
	return &url.URL{
		Scheme: fileScheme,
		Path:   path,
	}
}

// isWindowsDrivePath returns true if the file path is of the form used by
// Windows. We check if the path begins with a drive letter, followed by a ":".
func isWindowsDrivePath(path string) bool {
	if len(path) < 4 {
		return false
	}
	return unicode.IsLetter(rune(path[0])) && path[1] == ':'
}

// isWindowsDriveURI returns true if the file URI is of the format used by
// Windows URIs. The url.Parse package does not specially handle Windows paths
// (see https://github.com/golang/go/issues/6027). We check if the URI path has
// a drive prefix (e.g. "/C:"). If so, we trim the leading "/".
func isWindowsDriveURI(uri string) bool {
	if len(uri) < 4 {
		return false
	}
	return uri[0] == '/' && unicode.IsLetter(rune(uri[1])) && uri[2] == ':'
}
