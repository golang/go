// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cmd_go_bootstrap

// This code is compiled only into the bootstrap 'go' binary.
// These stubs avoid importing packages with large dependency
// trees, like the use of "net/http" in vcs.go.

package main

import (
	"errors"
	"io"
)

var errHTTP = errors.New("no http in bootstrap go command")

func httpGET(url string) ([]byte, error) {
	return nil, errHTTP
}

func httpsOrHTTP(importPath string) (string, io.ReadCloser, error) {
	return "", nil, errHTTP
}

func parseMetaGoImports(r io.Reader) (imports []metaImport) {
	panic("unreachable")
}
