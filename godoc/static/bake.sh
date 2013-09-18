#!/usr/bin/env bash
# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

STATIC="
	codewalk.html
	codewalkdir.html
	dirlist.html
	error.html
	example.html
	godoc.html
	godocs.js
	jquery.js
	opensearch.xml
	package.html
	package.txt
	play.js
	playground.js
	search.html
	search.txt
	style.css
"

go run bake.go $STATIC | gofmt > static.go
