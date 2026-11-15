// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"fmt"
	"internal/logosfile"
	"net/url"
	"path/filepath"
	"runtime"
	"strings"
)

func uriPath(uri string) (string, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return "", err
	}
	if u.Scheme != "file" {
		return "", fmt.Errorf("%q is not a file URI", uri)
	}
	if u.Host != "" && !strings.EqualFold(u.Host, "localhost") {
		return "", fmt.Errorf("%q is not a local file URI", uri)
	}
	p := u.Path
	if runtime.GOOS == "windows" && len(p) >= 3 && p[0] == '/' && p[2] == ':' {
		p = p[1:]
	}
	return filepath.Clean(filepath.FromSlash(p)), nil
}

func pathURI(path string) string {
	p := filepath.ToSlash(path)
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return (&url.URL{Scheme: "file", Path: p}).String()
}

func virtualPath(src string) string {
	base := filepath.Base(src)
	return filepath.Join(filepath.Dir(src), strings.TrimSuffix(base, filepath.Ext(base))+".go")
}

func isLogosPath(path string) bool {
	return logosfile.Is(path)
}

func isGoPath(path string) bool {
	return filepath.Ext(path) == ".go"
}
