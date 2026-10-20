// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"fmt"
	"net/url"
	"path/filepath"
	"runtime"
	"strings"

	"cmd/go/internal/cfg"
)

// uriPath returns the file path of a file: URI.
func uriPath(uri string) (string, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return "", err
	}
	if u.Scheme != "file" {
		return "", fmt.Errorf("%q is not a file URI", uri)
	}
	p := u.Path
	if runtime.GOOS == "windows" && len(p) >= 3 && p[0] == '/' && p[2] == ':' {
		p = p[1:]
	}
	return filepath.Clean(filepath.FromSlash(p)), nil
}

// pathURI returns the file: URI of an absolute path.
func pathURI(path string) string {
	p := filepath.ToSlash(path)
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return (&url.URL{Scheme: "file", Path: p}).String()
}

// virtualPath returns the path of the lowered Go document of a loGo/s
// source: the same directory, the file name with its loGo/s extension
// replaced by .go, which is the name the compiler gives the ejected file.
func virtualPath(src string) string {
	base := filepath.Base(src)
	return filepath.Join(filepath.Dir(src), strings.TrimSuffix(base, filepath.Ext(base))+".go")
}

// isLogosPath reports whether path names a loGo/s source file.
func isLogosPath(path string) bool {
	return cfg.IsLogosFile(filepath.Base(path))
}

// isGoPath reports whether path names a Go source file.
func isGoPath(path string) bool {
	return filepath.Ext(path) == ".go"
}
