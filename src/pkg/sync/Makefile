# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=sync
GOFILES=\
	mutex.go\
	rwmutex.go\

# 386-specific object files
OFILES_386=\
	asm_386.$O\

# amd64-specific object files
OFILES_amd64=\
	asm_amd64.$O\

GOARM?=6

# arm-specific object files
OFILES_arm=\
	asm_arm$(GOARM).$O\

OFILES=\
	$(OFILES_$(GOARCH))\

include ../../Make.pkg
