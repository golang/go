# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../Make.inc

all: install

# Only build tools for current architecture, and only tools written in C.
# The tools written in Go are managed by ../pkg/Makefile.
# We need all the C compilers for rebuilding generated files in runtime.
DIRS=\
	$(O)a\
	5c\
	6c\
	8c\
	$(O)g\
	$(O)l\
	cc\
	cov\
	gc\
	gopack\
	nm\
	prof\
	
# Clean applies to all directories, even for other architectures or 
# written in Go.
CLEANDIRS=\
	$(DIRS)\
	5a\
	5c\
	5g\
	5l\
	6a\
	6c\
	6g\
	6l\
	8a\
	8c\
	8g\
	8l\
	cgo\
	godoc\
	gofix\
	gofmt\
	goinstall\
	gotest\
	goyacc\
	hgpatch\

install: $(patsubst %,%.install,$(DIRS))
clean: $(patsubst %,%.clean,$(CLEANDIRS))
nuke: $(patsubst %,%.nuke,$(CLEANDIRS))

%.install:
	@echo
	@echo %%%% making $* %%%%
	@echo
	$(MAKE) -C $* install

gc.install 5c.install 6c.install 8c.install: cc.install
$(O)g.install: gc.install
$(O)a.install $(O)c.install $(O)g.install: $(O)l.install

%.clean:
	$(MAKE) -C $* clean

%.nuke:
	$(MAKE) -C $* nuke

echo-dirs:
	@echo $(DIRS)
