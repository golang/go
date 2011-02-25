# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=sync/atomic
GOFILES=\
	doc.go\

OFILES=\
	asm_$(GOARCH).$O\

ifeq ($(GOARCH),arm)
OFILES+=asm_$(GOOS)_$(GOARCH).$O
endif

include ../../../Make.pkg
