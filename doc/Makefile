# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

HTML=\
	articles/defer_panic_recover.html\
	articles/error_handling.html\
	articles/slices_usage_and_internals.html\
	effective_go.html\
	go1.html\

all: tmpltohtml $(HTML)

tmpltohtml: tmpltohtml.go
	go build tmpltohtml.go

%.html: %.tmpl tmpltohtml
	./makehtml $*.tmpl
