// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9

package gccgoimporter

var aliasTests = []importerTest{
	{pkgpath: "aliases", name: "A14", want: "type A14 = func(int, T0) chan T2"},
	{pkgpath: "aliases", name: "C0", want: "type C0 struct{f1 C1; f2 C1}"},
}

func init() {
	importerTests = append(importerTests, aliasTests...)
}
