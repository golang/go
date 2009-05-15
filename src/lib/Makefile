# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: install

GC=6g

DIRS=\
	bignum\
	bufio\
	container/vector\
	crypto/aes\
	exec\
	exvar\
	flag\
	fmt\
	go/ast\
	go/doc\
	go/parser\
	go/scanner\
	go/token\
	hash/adler32\
	hash/crc32\
	hash/md5\
	hash/sha1\
	hash/sum\
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
	sort\
	strconv\
	strings\
	sync\
	tabwriter\
	template\
	testing\
	time\
	unicode\
	utf8\

TEST=\
	bignum\
	bufio\
	container/vector\
	crypto/aes\
	exec\
	exvar\
	flag\
	fmt\
	go/parser\
	go/scanner\
	hash/adler32\
	hash/crc32\
	hash/md5\
	hash/sha1\
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

# TODO: dependencies - should auto-generate

bignum.install: fmt.install
bufio.install: io.install os.install
exec.install: os.install strings.install
exvar.install: fmt.install http.install log.install strconv.install sync.install
flag.install: fmt.install os.install strconv.install
log.install: fmt.install io.install os.install runtime.install time.install
path.install: io.install
once.install: sync.install
strings.install: utf8.install
testing.install: flag.install fmt.install runtime.install

fmt.install: io.install reflect.install strconv.install
GODEPS=fmt.install io.install once.install regexp.install sort.install \
	strconv.install strings.install unicode.install utf8.install container/vector.install
go/scanner.install: go/token.install $(GODEPS)
go/parser.install: go/ast.install go/scanner.install go/token.install $(GODEPS)
go/doc.install: go/ast.install go/token.install $(GODEPS)
go/ast.install: go/token.install $(GODEPS)
go/token.install: $(GODEPS)
hash/adler32.install: os.install
hash/crc32.install: os.install
hash/md5.install: os.install
hash/sha1.install: os.install
hash/sum.install: io.install os.install
http.install: bufio.install io.install net.install os.install path.install strings.install log.install
io.install: os.install sync.install
json.install: container/vector.install fmt.install io.install math.install \
	strconv.install strings.install utf8.install
# TODO(rsc): net is not supposed to depend on fmt or strings or strconv
net.install: fmt.install once.install os.install strconv.install strings.install
os.install: syscall.install once.install
regexp.install: os.install runtime.install
reflect.install: strconv.install sync.install
strconv.install: math.install os.install utf8.install
sync.install:
syscall.install: sync.install
tabwriter.install: os.install io.install container/vector.install
template.install: container/vector.install fmt.install io.install os.install reflect.install runtime.install strings.install
time.install: once.install os.install io.install

	once.install os.install sort.install strings.install sync.install template.install time.install \
