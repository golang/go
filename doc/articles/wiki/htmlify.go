// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io/ioutil"
	"os"
	"text/template"
)

func main() {
	b, _ := ioutil.ReadAll(os.Stdin)
	template.HTMLEscape(os.Stdout, b)
}
