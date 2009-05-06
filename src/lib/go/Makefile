# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: install

GC=6g

DIRS=\
	ast \
	doc \
	parser \
	scanner \
	token \


TEST=\
	parser \
	scanner \

clean.dirs: $(addsuffix .clean, $(DIRS))
install.dirs: $(addsuffix .install, $(DIRS))
test.dirs: $(addsuffix .test, $(TEST))

%.clean:
	+cd $* && make clean

%.install:
	+cd $* && make install

%.test:
	+cd $* && make test

clean: clean.dirs

install: install.dirs

test:	test.dirs

# TODO: dependencies - should auto-generate

scanner.install: token.install
parser.install:	ast.install scanner.install token.install
doc.install: ast.install token.install
ast.install: token.install
