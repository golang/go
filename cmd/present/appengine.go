// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package main

import (
	"mime"

	"golang.org/x/tools/present"
)

func init() {
	initTemplates("./present/")
	present.PlayEnabled = true
	initPlayground("./present/", nil)

	// App Engine has no /etc/mime.types
	mime.AddExtensionType(".svg", "image/svg+xml")
}
