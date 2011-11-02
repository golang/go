# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# After editing the DIRS= list or adding imports to any Go files
# in any of those directories, run:
#
#	./deps.bash
#
# to rebuild the dependency information in Make.deps.

include ../Make.inc

all: install

DIRS=\
	archive/tar\
	archive/zip\
	asn1\
	big\
	bufio\
	bytes\
	cmath\
	compress/bzip2\
	compress/flate\
	compress/gzip\
	compress/lzw \
	compress/zlib\
	container/heap\
	container/list\
	container/ring\
	crypto\
	crypto/aes\
	crypto/blowfish\
	crypto/bcrypt\
	crypto/cast5\
	crypto/cipher\
	crypto/des\
	crypto/dsa\
	crypto/ecdsa\
	crypto/elliptic\
	crypto/hmac\
	crypto/md4\
	crypto/md5\
	crypto/ocsp\
	crypto/openpgp\
	crypto/openpgp/armor\
	crypto/openpgp/elgamal\
	crypto/openpgp/error\
	crypto/openpgp/packet\
	crypto/openpgp/s2k\
	crypto/rand\
	crypto/rc4\
	crypto/ripemd160\
	crypto/rsa\
	crypto/sha1\
	crypto/sha256\
	crypto/sha512\
	crypto/subtle\
	crypto/tls\
	crypto/twofish\
	crypto/x509\
	crypto/x509/pkix\
	crypto/xtea\
	csv\
	debug/dwarf\
	debug/macho\
	debug/elf\
	debug/gosym\
	debug/pe\
	encoding/ascii85\
	encoding/base32\
	encoding/base64\
	encoding/binary\
	encoding/git85\
	encoding/hex\
	encoding/pem\
	errors\
	exec\
	exp/ebnf\
	exp/ebnflint\
	exp/gotype\
	exp/gui\
	exp/gui/x11\
	exp/norm\
	exp/spdy\
	exp/sql\
	exp/sql/driver\
	exp/template/html\
	exp/types\
	expvar\
	flag\
	fmt\
	go/ast\
	go/build\
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
	hash/fnv\
	html\
	http\
	http/cgi\
	http/fcgi\
	http/pprof\
	http/httptest\
	image\
	image/bmp\
	image/color\
	image/draw\
	image/gif\
	image/jpeg\
	image/png\
	image/tiff\
	image/ycbcr\
	index/suffixarray\
	io\
	io/ioutil\
	json\
	log\
	mail\
	math\
	mime\
	mime/multipart\
	net\
	net/dict\
	net/textproto\
	old/netchan\
	old/regexp\
	old/template\
	os\
	os/signal\
	os/user\
	patch\
	path\
	path/filepath\
	rand\
	reflect\
	regexp\
	regexp/syntax\
	rpc\
	rpc/jsonrpc\
	runtime\
	runtime/cgo\
	runtime/debug\
	runtime/pprof\
	scanner\
	smtp\
	sort\
	strconv\
	strings\
	sync\
	sync/atomic\
	syscall\
	syslog\
	tabwriter\
	template\
	template/parse\
	testing\
	testing/iotest\
	testing/quick\
	testing/script\
	time\
	unicode\
	url\
	utf16\
	utf8\
	websocket\
	xml\
	../cmd/cgo\
	../cmd/godoc\
	../cmd/gofix\
	../cmd/gofmt\
	../cmd/goinstall\
	../cmd/gotest\
	../cmd/govet\
	../cmd/goyacc\
	../cmd/hgpatch\

ifeq ($(GOOS),linux)
DIRS+=\
	exp/inotify\

endif

ifeq ($(GOOS),plan9)
NOPLAN9BUILD=\
	os/signal\

DIRS:=$(filter-out $(NOPLAN9BUILD),$(DIRS))
endif

NOTEST+=\
	crypto\
	crypto/openpgp/error\
	crypto/x509/pkix\
	exp/ebnflint\
	exp/gui\
	exp/gui/x11\
	exp/sql/driver\
	go/doc\
	hash\
	http/pprof\
	http/httptest\
	image/bmp\
	image/color\
	image/gif\
	net/dict\
	rand\
	runtime/cgo\
	syscall\
	testing\
	testing/iotest\
	../cmd/cgo\
	../cmd/godoc\
	../cmd/gotest\
	../cmd/goyacc\
	../cmd/hgpatch\

NOBENCH+=\

# Disable tests that windows cannot run yet.
ifeq ($(GOOS),windows)
NOTEST+=os/signal    # no signals
NOTEST+=syslog       # no network
endif

TEST=\
	$(filter-out $(NOTEST),$(DIRS))

BENCH=\
	$(filter-out $(NOBENCH),$(TEST))

clean.dirs: $(addsuffix .clean, $(DIRS))
install.dirs: $(addsuffix .install, $(DIRS))
nuke.dirs: $(addsuffix .nuke, $(DIRS))
test.dirs: $(addsuffix .test, $(TEST))
testshort.dirs: $(addsuffix .testshort, $(TEST))
bench.dirs: $(addsuffix .bench, $(BENCH))

%.clean:
	+$(MAKE) -C $* clean

%.install:
	+@echo install $*
	+@$(MAKE) -C $* install.clean >$*/build.out 2>&1 || (echo INSTALL FAIL $*; cat $*/build.out; exit 1)

%.nuke:
	+$(MAKE) -C $* nuke

%.test:
	+@echo test $*
	+@$(MAKE) -C $* test.clean >$*/test.out 2>&1 || (echo TEST FAIL $*; cat $*/test.out; exit 1)

%.testshort:
	+@echo test $*
	+@$(MAKE) -C $* testshort.clean >$*/test.out 2>&1 || (echo TEST FAIL $*; cat $*/test.out; exit 1)

%.bench:
	+$(MAKE) -C $* bench	

clean: clean.dirs

install: install.dirs

test:	test.dirs

testshort: testshort.dirs

bench:	bench.dirs ../../test/garbage.bench

nuke: nuke.dirs
	rm -rf "$(GOROOT)"/pkg/*

deps:
	./deps.bash

echo-dirs:
	@echo $(DIRS)

-include Make.deps

runtime/cgo.install: ../cmd/cgo.install
