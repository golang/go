# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

# The directory is prof because the source is portable and general.
# We call the binary 6prof to avoid confusion and because this binary
# is linked only with amd64 and x86 support.

TARG=6prof
OFILES=\
	main.$O\

#HFILES=\
#	defs.h\
#	fns.h\

$(TARG): $(OFILES)
	$(LD) -o $(TARG) -L"$(GOROOT)"/lib $(OFILES) -lmach -lbio -l9

clean:
	rm -f *.$O $(TARG)

ifeq ($(GOOS),windows)
NAME=windows
else
NAME=$(shell uname | tr A-Z a-z)
endif

install: install-$(NAME) install-pprof
install-linux: install-default
install-freebsd: install-default
install-windows: install-default

# on Darwin, have to install and setgid; see $GOROOT/src/sudo.bash
install-darwin: $(TARG)
	@true

install-default: $(TARG)
	cp $(TARG) "$(GOBIN)"/$(TARG)

$(OFILES): $(HFILES)

install-pprof: gopprof
	cp gopprof "$(GOBIN)"/gopprof
