# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# After editing the DIRS= list or adding imports to any Go files
# in any of those directories, run:
#
#	./deps.bash
#
# to rebuild the dependency information in Make.deps.

nullstring :=
space := $(nullstring) # a space at the end
ifndef GOBIN
QUOTED_HOME=$(subst $(space),\ ,$(HOME))
GOBIN=$(QUOTED_HOME)/bin
endif
QUOTED_GOBIN=$(subst $(space),\ ,$(GOBIN))

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
	crypto/blowfish\
	crypto/hmac\
	crypto/md4\
	crypto/md5\
	crypto/rc4\
	crypto/rsa\
	crypto/sha1\
	crypto/sha256\
	crypto/subtle\
	crypto/tls\
	crypto/x509\
	crypto/xtea\
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
	encoding/hex\
	encoding/pem\
	exec\
	exp/datafmt\
	exp/draw\
	exp/eval\
	exp/exception\
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
	image/jpeg\
	image/png\
	io\
	io/ioutil\
	json\
	log\
	math\
	mime\
	net\
	once\
	os\
	os/signal\
	patch\
	path\
	rand\
	reflect\
	regexp\
	rpc\
	runtime\
	scanner\
	sort\
	strconv\
	strings\
	sync\
	syscall\
	syslog\
	tabwriter\
	template\
	testing\
	testing/iotest\
	testing/quick\
	testing/script\
	time\
	unicode\
	utf8\
	websocket\
	xgb\
	xml\

NOTEST=\
	debug/proc\
	go/ast\
	go/doc\
	go/token\
	hash\
	image\
	image/jpeg\
	rand\
	runtime\
	syscall\
	testing/iotest\
	xgb\

NOBENCH=\
	container/vector\

TEST=\
	$(filter-out $(NOTEST),$(DIRS))

BENCH=\
	$(filter-out $(NOBENCH),$(TEST))

clean.dirs: $(addsuffix .clean, $(DIRS))
install.dirs: $(addsuffix .install, $(DIRS))
nuke.dirs: $(addsuffix .nuke, $(DIRS))
test.dirs: $(addsuffix .test, $(TEST))
bench.dirs: $(addsuffix .bench, $(BENCH))

%.clean:
	+cd $* && $(QUOTED_GOBIN)/gomake clean

%.install:
	+cd $* && $(QUOTED_GOBIN)/gomake install

%.nuke:
	+cd $* && $(QUOTED_GOBIN)/gomake nuke

%.test:
	+cd $* && $(QUOTED_GOBIN)/gomake test

%.bench:
	+cd $* && $(QUOTED_GOBIN)/gomake bench

clean: clean.dirs

install: install.dirs

test:	test.dirs

bench:	bench.dirs ../../test/garbage.bench

nuke: nuke.dirs
	rm -rf "$(GOROOT)"/pkg/*

deps:
	./deps.bash

-include Make.deps
