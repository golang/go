// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package main

import (
	"mime"

	"code.google.com/p/go.tools/present"

	_ "code.google.com/p/go.tools/playground"
)

var basePath = "./present/"

func init() {
	initTemplates(basePath)
	playScript(basePath, "HTTPTransport")
	present.PlayEnabled = true

	// App Engine has no /etc/mime.types
	mime.AddExtensionType(".svg", "image/svg+xml")
}

func playable(c present.Code) bool {
	return present.PlayEnabled && c.Play && c.Ext == ".go"
}
