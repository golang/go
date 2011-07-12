# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=hash/crc32

ifeq ($(GOARCH), amd64)
	ARCH_GOFILES=crc32_amd64.go
	OFILES=crc32_amd64.6
else
	ARCH_GOFILES=crc32_generic.go
endif

GOFILES=\
	crc32.go\
	$(ARCH_GOFILES)

include ../../../Make.pkg
