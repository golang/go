// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cmd_go_bootstrap

// This code is compiled only into the bootstrap 'go' binary.
// These stubs avoid importing packages with large dependency
// trees, like the use of "net/http" in vcs.go.

package web

import (
	"errors"
	"io"
)

var errHTTP = errors.New("no http in bootstrap go command")

type HTTPError struct {
	StatusCode int
}

func (e *HTTPError) Error() string {
	panic("unreachable")
}

func Get(url string) ([]byte, error) {
	return nil, errHTTP
}

func GetMaybeInsecure(importPath string, security SecurityMode) (string, io.ReadCloser, error) {
	return "", nil, errHTTP
}

func QueryEscape(s string) string { panic("unreachable") }
func OpenBrowser(url string) bool { panic("unreachable") }
