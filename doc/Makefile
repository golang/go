# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../src/Make.inc

TARG=tmpltohtml
GOFILES=\
	tmpltohtml.go\

HTML=\
	articles/defer_panic_recover.html\
	articles/error_handling.html\
	effective_go.html\
	go1.html\
	go_tutorial.html\

all: tmpltohtml $(HTML)

%.html: %.tmpl tmpltohtml
	./makehtml $*.tmpl

include ../src/Make.cmd
