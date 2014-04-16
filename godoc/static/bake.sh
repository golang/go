#!/usr/bin/env bash
# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

STATIC="
	analysis/call3.png
	analysis/call-eg.png
	analysis/callers1.png
	analysis/callers2.png
	analysis/chan1.png
	analysis/chan2a.png
	analysis/chan2b.png
	analysis/error1.png
	analysis/help.html
	analysis/ident-def.png
	analysis/ident-field.png
	analysis/ident-func.png
	analysis/ipcg-func.png
	analysis/ipcg-pkg.png
	analysis/typeinfo-pkg.png
	analysis/typeinfo-src.png
	callgraph.html
	codewalk.html
	codewalkdir.html
	dirlist.html
	error.html
	example.html
	godoc.html
	godocs.js
	images/minus.gif
	images/plus.gif
	images/treeview-black-line.gif
	images/treeview-black.gif
	images/treeview-default-line.gif
	images/treeview-default.gif
	images/treeview-gray-line.gif
	images/treeview-gray.gif
	implements.html
	jquery.js
	jquery.treeview.css
	jquery.treeview.edit.js
	jquery.treeview.js
	methodset.html
	opensearch.xml
	package.html
	package.txt
	play.js
	playground.js
	search.html
	search.txt
	searchcode.html
	searchdoc.html
	searchtxt.html
	style.css
"

go run bake.go $STATIC | gofmt > static.go
