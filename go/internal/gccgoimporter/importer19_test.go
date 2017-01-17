// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9

package gccgoimporter

var aliasTests = []importerTest{
	{pkgpath: "alias", name: "IntAlias2", want: "type IntAlias2 = Int"},
}

func init() {
	importerTests = append(importerTests, aliasTests...)
}
