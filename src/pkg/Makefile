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
	bufio\
	bytes\
	cmath\
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
	crypto/rand\
	crypto/rc4\
	crypto/ripemd160\
	crypto/rsa\
	crypto/sha1\
	crypto/sha256\
	crypto/sha512\
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
	exp/bignum\
	exp/datafmt\
	exp/draw\
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
	hash/crc64\
	http\
	http/pprof\
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
	netchan\
	nntp\
	once\
	os\
	os/signal\
	patch\
	path\
	rand\
	reflect\
	regexp\
	rpc\
	rpc/jsonrpc\
	runtime\
	runtime/pprof\
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
	utf16\
	utf8\
	websocket\
	xml\

NOTEST=\
	debug/proc\
	go/ast\
	go/doc\
	go/token\
	hash\
	http/pprof\
	image\
	image/jpeg\
	rand\
	runtime\
	runtime/pprof\
	syscall\
	testing/iotest\

NOBENCH=\
	container/vector\

# Disable tests that depend on an external network.
ifeq ($(DISABLE_NET_TESTS),1)
NOTEST+=http net
endif

# Disable tests that NaCl cannot run yet.
ifeq ($(GOOS),nacl)
NOTEST+=archive/tar  # no pipe
NOTEST+=debug/dwarf  # no pread
NOTEST+=debug/macho  # no pread
NOTEST+=debug/elf    # no pread
NOTEST+=exec         # no pipe
NOTEST+=http         # no network
NOTEST+=log          # no runtime.Caller
NOTEST+=net          # no network
NOTEST+=netchan      # no network
NOTEST+=os           # many things unimplemented
NOTEST+=os/signal    # no signals
NOTEST+=path         # tree walking does not work
NOTEST+=rpc          # no network
NOTEST+=syslog       # no network
NOTEST+=time         # no syscall.Kill, syscall.SIGCHLD for sleep tests
NOTEST+=websocket    # no network
endif

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
