# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

# The directory is cov because the source is portable and general.
# We call the binary 6cov to avoid confusion and because this binary
# is linked only with amd64 and x86 support.

TARG=cov
OFILES=\
	main.$O\
	tree.$O\

HFILES=\
	tree.h\

NOINSTALL=1
include ../../Make.ctool

ifeq ($(GOOS),windows)
NAME=windows
else
NAME=$(shell uname | tr A-Z a-z)
endif

install: install-$(NAME)
install-linux: install-default
install-freebsd: install-default
install-netbsd: install-default
install-openbsd: install-default
install-windows: install-default

# on Darwin, have to install and setgid; see $GOROOT/src/sudo.bash
install-darwin: $(TARG)
	@true

install-default: $(TARG)
	cp $(TARG) "$(GOROOT)"/bin/go-tool/$(TARG)
