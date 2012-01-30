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

GOOS_LIST=\
	darwin\
	freebsd\
	linux\
	netbsd\
	openbsd\
	plan9\
	windows\

GOARCH_LIST=\
	386\
	amd64\
	arm\

ifeq ($(filter $(GOOS),$(GOOS_LIST)),)
$(error Invalid $$GOOS '$(GOOS)'; must be one of: $(GOOS_LIST))
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

ifeq ($(filter $(GOARCH),$(GOARCH_LIST)),)
$(error Invalid $$GOARCH '$(GOARCH)'; must be one of: $(GOARCH_LIST))
endif

ifeq ($(GOARCH),386)
O:=8
else ifeq ($(GOARCH),amd64)
O:=6
else ifeq ($(GOARCH),arm)
O:=5
ifneq ($(GOOS),linux)
$(error Invalid $$GOOS '$(GOOS)' for GOARCH=arm; must be linux)
endif
else
$(error Missing $$O for '$(GOARCH)')
endif

# Save for recursive make to avoid recomputing.
export GOARCH GOOS GOHOSTARCH GOHOSTOS GOARCH_LIST GOOS_LIST

# ugly hack to deal with whitespaces in $GOROOT
nullstring :=
space := $(nullstring) # a space at the end
QUOTED_GOROOT:=$(subst $(space),\ ,$(GOROOT))

# default GOBIN
ifndef GOBIN
GOBIN=$(QUOTED_GOROOT)/bin
endif
QUOTED_GOBIN=$(subst $(space),\ ,$(GOBIN))

AS=$(QUOTED_GOROOT)/bin/go-tool/$(O)a
CC=$(QUOTED_GOROOT)/bin/go-tool/$(O)c
GC=$(QUOTED_GOROOT)/bin/go-tool/$(O)g
LD=$(QUOTED_GOROOT)/bin/go-tool/$(O)l
OS=568vq
CFLAGS=-FVw

HOST_CC=quietgcc
HOST_LD=quietgcc
HOST_O=o
HOST_YFLAGS=-d
HOST_AR?=ar

# These two variables can be overridden in the environment
# to build with other flags.  They are like $CFLAGS and $LDFLAGS
# in a more typical GNU build.  We are more explicit about the names
# here because there are different compilers being run during the
# build (both gcc and 6c, for example).
HOST_EXTRA_CFLAGS?=-ggdb -O2
HOST_EXTRA_LDFLAGS?=

# The -fno-common here is not necessary, but some compilers
# on OS X seem to set it by default.  Setting it here keeps the build
# working in that non-standard context.
HOST_CFLAGS=-fno-common -I"$(GOROOT)/include" $(HOST_EXTRA_CFLAGS)
HOST_LDFLAGS=$(HOST_EXTRA_LDFLAGS)
PWD=$(shell pwd)

# Decide whether use of cgo is okay.
ifeq ($(CGO_ENABLED),)
# Default on...
CGO_ENABLED:=1
ifeq ($(GOARCH),arm)  # ... but not on ARM
CGO_ENABLED:=0
endif
ifeq ($(GOOS),plan9)  # ... and not on Plan 9
CGO_ENABLED:=0
endif
ifeq ($(GOOS),openbsd)  # ... and not on OpenBSD
CGO_ENABLED:=0
endif
ifeq ($(GOOS),netbsd)  # ... and not on NetBSD
CGO_ENABLED:=0
endif
endif

# Make environment more standard.
LANG:=
LC_ALL:=C
LC_CTYPE:=C
GREP_OPTIONS:=
GREP_COLORS:=
export LANG LC_ALL LC_CTYPE GREP_OPTIONS GREP_COLORS

go-env:
	@echo export GOARCH="$(GOARCH)"
	@echo export GOOS="$(GOOS)"
	@echo export GOHOSTARCH="$(GOHOSTARCH)"
	@echo export GOHOSTOS="$(GOHOSTOS)"
	@echo export CGO_ENABLED="$(CGO_ENABLED)"
	@echo export O="$O"
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
