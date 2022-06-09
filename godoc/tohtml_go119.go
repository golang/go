// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package godoc

import (
	"bytes"
	"go/doc"
)

func godocToHTML(buf *bytes.Buffer, pkg *doc.Package, comment string) {
	buf.Write(pkg.HTML(comment))
}
