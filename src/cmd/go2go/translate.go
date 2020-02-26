// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/go2go"
	"io/ioutil"
	"strings"
)

// translate writes .go files for all .go2 files in dir.
func translate(importer *go2go.Importer, dir string) {
	if err := go2go.Rewrite(importer, dir); err != nil {
		die(err.Error())
	}
}

// translateFile translates one .go2 file into a .go file.
func translateFile(importer *go2go.Importer, file string) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		die(err.Error())
	}
	out, err := go2go.RewriteBuffer(importer, file, data)
	if err != nil {
		die(err.Error())
	}
	if err := ioutil.WriteFile(strings.TrimSuffix(file, ".go2") + ".go", out, 0644); err != nil {
		die(err.Error())
	}
}
