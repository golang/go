# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# After editing the DIRS= list or adding imports to any Go files
# in any of those directories, run:
#
#	./deps.bash
#
# to rebuild the dependency information in Make.deps.

all: install

DIRS=\
	bignum\
	bufio\
	bytes\
	container/list\
	container/vector\
	crypto/aes\
	crypto/block\
	crypto/hmac\
	crypto/md5\
	crypto/sha1\
	datafmt\
	exec\
	exvar\
	flag\
	fmt\
	go/ast\
	go/doc\
	go/parser\
	go/scanner\
	go/token\
	hash\
	hash/adler32\
	hash/crc32\
	http\
	io\
	json\
	log\
	malloc\
	math\
	net\
	once\
	os\
	path\
	rand\
	reflect\
	regexp\
	runtime\
	sort\
	strconv\
	strings\
	sync\
	syscall\
	tabwriter\
	template\
	testing\
	testing/iotest\
	time\
	unicode\
	utf8\

TEST=\
	bignum\
	bufio\
	container/list\
	container/vector\
	crypto/aes\
	crypto/block\
	crypto/md5\
	crypto/sha1\
	datafmt\
	exec\
	exvar\
	flag\
	fmt\
	go/parser\
	go/scanner\
	hash/adler32\
	hash/crc32\
	http\
	io\
	json\
	log\
	math\
	net\
	once\
	os\
	path\
	reflect\
	regexp\
	sort\
	strconv\
	strings\
	sync\
	tabwriter\
	template\
	time\
	unicode\
	utf8\

clean.dirs: $(addsuffix .clean, $(DIRS))
install.dirs: $(addsuffix .install, $(DIRS))
nuke.dirs: $(addsuffix .nuke, $(DIRS))
test.dirs: $(addsuffix .test, $(TEST))

%.clean:
	+cd $* && make clean

%.install:
	+cd $* && make install

%.nuke:
	+cd $* && make nuke

%.test:
	+cd $* && make test

clean: clean.dirs

install: install.dirs

test:	test.dirs

nuke: nuke.dirs
	rm -rf $(GOROOT)/pkg/*

deps:
	./deps.bash

include Make.deps
