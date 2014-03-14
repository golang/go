#!/usr/bin/env bash
# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

STATIC="
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
