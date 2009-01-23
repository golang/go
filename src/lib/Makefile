# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: install

GC=6g

DIRS=\
	container/array\
	fmt\
	hash\
	http\
	io\
	json\
	math\
	net\
	os\
	reflect\
	regexp\
	strconv\
	sync\
	tabwriter\
	time\
	unicode\

FILES=\
	bufio\
	flag\
	malloc\
	once\
	rand\
	sort\
	strings\
	testing\
	utf8\
	bignum\

TEST=\
	bufio\
	once\
	sort\
	strings\
	utf8\
	bignum\

clean.dirs: $(addsuffix .dirclean, $(DIRS))
install.dirs: $(addsuffix .dirinstall, $(DIRS))
install.files: $(addsuffix .install, $(FILES))
nuke.dirs: $(addsuffix .dirnuke, $(DIRS))
test.files: $(addsuffix .test, $(TEST))

%.6: container/%.go
	$(GC) container/$*.go

%.6: %.go
	$(GC) $*.go

%.test: %.6
	gotest $*_test.go

%.clean:
	rm -f $*.6

%.install: %.6
	mv $*.6 $(GOROOT)/pkg/$*.6

%.dirclean:
	+cd $* && make clean

%.dirinstall:
	+cd $* && make install

%.dirnuke:
	+cd $* && make nuke

clean.files:
	rm -f 6.out *.6

clean: clean.dirs clean.files

install: install.dirs install.files

nuke: nuke.dirs clean.files
	rm -f $(GOROOT)/pkg/*

test: test.files

# TODO: dependencies - should auto-generate

bignum.6: fmt.dirinstall
bufio.6: io.dirinstall os.dirinstall
flag.6: fmt.dirinstall
testing.6: flag.install fmt.dirinstall
strings.6: utf8.install

fmt.dirinstall: io.dirinstall reflect.dirinstall strconv.dirinstall
hash.dirinstall: os.dirinstall
http.dirinstall: bufio.install io.dirinstall net.dirinstall os.dirinstall strings.install
io.dirinstall: os.dirinstall syscall.dirinstall
json.dirinstall: container/array.dirinstall fmt.dirinstall io.dirinstall math.dirinstall \
	strconv.dirinstall strings.install utf8.install
# TODO(rsc): net is not supposed to depend on fmt or strings or strconv
net.dirinstall: fmt.dirinstall once.install os.dirinstall strconv.dirinstall strings.install
os.dirinstall: syscall.dirinstall
regexp.dirinstall: os.dirinstall
reflect.dirinstall: strconv.dirinstall sync.dirinstall
strconv.dirinstall: math.dirinstall os.dirinstall utf8.install
tabwriter.dirinstall: os.dirinstall io.dirinstall container/array.dirinstall
time.dirinstall: once.install os.dirinstall
sync.dirinstall:

