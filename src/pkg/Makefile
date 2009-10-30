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
	archive/tar\
	asn1\
	big\
	bignum\
	bufio\
	bytes\
	compress/flate\
	compress/gzip\
	compress/zlib\
	container/heap\
	container/list\
	container/ring\
	container/vector\
	crypto/aes\
	crypto/block\
	crypto/hmac\
	crypto/md5\
	crypto/rc4\
	crypto/sha1\
	debug/dwarf\
	debug/macho\
	debug/elf\
	debug/gosym\
	debug/proc\
	ebnf\
	encoding/ascii85\
	encoding/base64\
	encoding/binary\
	encoding/git85\
	encoding/pem\
	exec\
	exp/datafmt\
	exp/eval\
	exp/iterable\
	expvar\
	flag\
	fmt\
	go/ast\
	go/doc\
	go/parser\
	go/printer\
	go/scanner\
	go/token\
	gob\
	hash\
	hash/adler32\
	hash/crc32\
	http\
	image\
	image/png\
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
	rpc\
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
	testing/quick\
	time\
	unicode\
	utf8\
	xml\

NOTEST=\
	debug/proc\
	go/ast\
	go/doc\
	go/token\
	hash\
	image\
	malloc\
	rand\
	runtime\
	syscall\
	testing/iotest\

TEST=\
	$(filter-out $(NOTEST),$(DIRS))

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
