# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../src/Make.inc

TARG=tmpltohtml
GOFILES=\
	tmpltohtml.go\

all: tmpltohtml go_tutorial.html effective_go.html

go_tutorial.html: go_tutorial.tmpl tmpltohtml
	./makehtml go_tutorial.tmpl

effective_go.html: effective_go.tmpl tmpltohtml
	./makehtml effective_go.tmpl

include ../src/Make.cmd
