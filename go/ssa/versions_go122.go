// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.22
// +build go1.22

package ssa

import (
	"go/ast"
)

func init() {
	fileVersions = func(file *ast.File) string {
		if maj, min := parseGoVersion(file.GoVersion); maj >= 0 && min >= 0 {
			return file.GoVersion
		}
		return ""
	}
}
