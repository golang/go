# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Makefile included by all other Go makefiles.

# Clear variables that must come from Makefiles,
# not the environment.
LIB:=
TARG:=
GOFILES:=
HFILES:=
OFILES:=
YFILES:=

# GOROOT must be set.
ifeq ($(GOROOT),)
$(error $$GOROOT is not set; use gomake or set $$GOROOT in your environment)
endif

# Set up GOROOT_FINAL, GOARCH, GOOS if needed.
GOROOT_FINAL?=$(GOROOT)

ifeq ($(GOHOSTOS),)
GOHOSTOS:=$(shell uname | tr A-Z a-z | sed 's/mingw/windows/; s/.*windows.*/windows/')
endif

ifeq ($(GOOS),)
GOOS:=$(GOHOSTOS)
endif

ifeq ($(GOOS),darwin)
else ifeq ($(GOOS),freebsd)
else ifeq ($(GOOS),linux)
else ifeq ($(GOOS),nacl)
else ifeq ($(GOOS),tiny)
else ifeq ($(GOOS),windows)
else
$(error Invalid $$GOOS '$(GOOS)'; must be darwin, freebsd, linux, nacl, tiny, or windows)
endif

ifeq ($(GOHOSTARCH),)
ifeq ($(GOHOSTOS),darwin)
# Even on 64-bit platform, darwin uname -m prints i386.
# Check for amd64 with sysctl instead.
GOHOSTARCH:=${shell if sysctl machdep.cpu.extfeatures | grep EM64T >/dev/null; then echo amd64; else uname -m | sed 's/i386/386/'; fi}
else
# Ask uname -m for the processor.
GOHOSTARCH:=${shell uname -m | sed 's/^..86$$/386/; s/^.86$$/386/; s/x86_64/amd64/; s/arm.*/arm/'}
endif
endif

ifeq ($(GOARCH),)
GOARCH:=$(GOHOSTARCH)
endif

# darwin requires GOHOSTARCH match GOARCH
ifeq ($(GOOS),darwin)
GOHOSTARCH:=$(GOARCH)
endif

ifeq ($(GOARCH),386)
O:=8
else ifeq ($(GOARCH),amd64)
O:=6
else ifeq ($(GOARCH),arm)

O:=5
ifeq ($(GOOS),linux)
else
$(error Invalid $$GOOS '$(GOOS)' for GOARCH=arm; must be linux)
endif

else
$(error Invalid $$GOARCH '$(GOARCH)'; must be 386, amd64, or arm)
endif

# Save for recursive make to avoid recomputing.
export GOARCH GOOS GOHOSTARCH GOHOSTOS

# ugly hack to deal with whitespaces in $GOROOT
nullstring :=
space := $(nullstring) # a space at the end
QUOTED_GOROOT:=$(subst $(space),\ ,$(GOROOT))

# default GOBIN
ifndef GOBIN
GOBIN=$(QUOTED_GOROOT)/bin
endif
QUOTED_GOBIN=$(subst $(space),\ ,$(GOBIN))

AS=${O}a
CC=${O}c
GC=${O}g
LD=${O}l
OS=568vq
CFLAGS=-FVw

HOST_CC=quietgcc
HOST_LD=quietgcc
HOST_O=o
HOST_YFLAGS=-d
HOST_CFLAGS=-ggdb -I"$(GOROOT)/include" -O2 -fno-inline
PWD=$(shell pwd)

# Make environment more standard.
LANG:=
LC_ALL:=C
LC_CTYPE:=C
GREP_OPTIONS:=
GREP_COLORS:=
export LANG LC_ALL LC_CTYPE GREP_OPTIONS GREP_COLORS

go-env:
	@echo export GOARCH=$(GOARCH)
	@echo export GOOS=$(GOOS)
	@echo export GOHOSTARCH=$(GOHOSTARCH)
	@echo export GOHOSTOS=$(GOHOSTOS)
	@echo export O=$O
	@echo export AS="$(AS)"
	@echo export CC="$(CC)"
	@echo export GC="$(GC)"
	@echo export LD="$(LD)"
	@echo export OS="$(OS)"
	@echo export CFLAGS="$(CFLAGS)"
	@echo export LANG="$(LANG)"
	@echo export LC_ALL="$(LC_ALL)"
	@echo export LC_CTYPE="$(LC_CTYPE)"
	@echo export GREP_OPTIONS="$(GREP_OPTIONS)"
	@echo export GREP_COLORS="$(GREP_COLORS)"
	@echo MAKE_GO_ENV_WORKED=1

# Don't let the targets in this file be used
# as the default make target.
.DEFAULT_GOAL:=
