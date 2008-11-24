# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: install

GC=6g

DIRS=\
	container/array\
	fmt\
	http\
	math\
	net\
	os\
	reflect\
	regexp\
	strconv\
	tabwriter\
	time\

FILES=\
	bignum\
	bufio\
	vector\
	flag\
	io\
	once\
	rand\
	sort\
	strings\
	testing\
	utf8\

TEST=\
	utf8\

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
bufio.6: io.install os.dirinstall
flag.6: fmt.dirinstall
io.6: os.dirinstall syscall.dirinstall
testing.6: flag.install fmt.dirinstall

fmt.dirinstall: io.install reflect.dirinstall strconv.dirinstall
http.dirinstall: bufio.install io.install net.dirinstall os.dirinstall strings.install
net.dirinstall: once.install os.dirinstall strconv.dirinstall
os.dirinstall: syscall.dirinstall
regexp.dirinstall: os.dirinstall
reflect.dirinstall: strconv.dirinstall
strconv.dirinstall: os.dirinstall utf8.install
tabwriter.dirinstall: os.dirinstall io.install container/array.dirinstall
time.dirinstall: once.install os.dirinstall

