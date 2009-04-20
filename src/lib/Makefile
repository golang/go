# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: install

GC=6g

DIRS=\
	container\
	fmt\
	go\
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
	template\
	time\
	unicode\

FILES=\
	bignum\
	bufio\
	exec\
	exvar\
	flag\
	log\
	malloc\
	once\
	rand\
	sort\
	strings\
	testing\
	utf8\

TEST=\
	bignum\
	bufio\
	exec\
	exvar\
	flag\
	log\
	once\
	sort\
	strings\
	utf8\

clean.dirs: $(addsuffix .dirclean, $(DIRS))
install.dirs: $(addsuffix .dirinstall, $(DIRS))
install.files: $(addsuffix .install, $(FILES))
nuke.dirs: $(addsuffix .dirnuke, $(DIRS))
test.files: $(addsuffix .test, $(TEST))

%.6: %.go
	$(GC) $*.go

%.test: %.6
	gotest $*_test.go

%.clean:
	rm -f $*.6

%.install: %.6
	6ar grc $*.a $*.6
	mv $*.a $(GOROOT)/pkg/$*.a
	rm -f $*.6

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
exec.6: os.dirinstall strings.install
exvar.6: fmt.dirinstall sync.dirinstall
flag.6: fmt.dirinstall os.dirinstall strconv.dirinstall
log.6: fmt.dirinstall io.dirinstall os.dirinstall time.dirinstall
path.6: io.dirinstall
once.6: sync.dirinstall
strings.6: utf8.install
testing.6: flag.install fmt.dirinstall

fmt.dirinstall: io.dirinstall reflect.dirinstall strconv.dirinstall
go.dirinstall: strconv.dirinstall utf8.install unicode.dirinstall fmt.dirinstall
hash.dirinstall: os.dirinstall
http.dirinstall: bufio.install io.dirinstall net.dirinstall os.dirinstall path.install strings.install log.install
io.dirinstall: os.dirinstall sync.dirinstall
json.dirinstall: container.dirinstall fmt.dirinstall io.dirinstall math.dirinstall \
	strconv.dirinstall strings.install utf8.install
# TODO(rsc): net is not supposed to depend on fmt or strings or strconv
net.dirinstall: fmt.dirinstall once.install os.dirinstall strconv.dirinstall strings.install
os.dirinstall: syscall.dirinstall once.install
regexp.dirinstall: os.dirinstall
reflect.dirinstall: strconv.dirinstall sync.dirinstall
strconv.dirinstall: math.dirinstall os.dirinstall utf8.install
sync.dirinstall:
syscall.dirinstall: sync.dirinstall
tabwriter.dirinstall: os.dirinstall io.dirinstall container.dirinstall
template.dirinstall: fmt.dirinstall io.dirinstall os.dirinstall reflect.dirinstall strings.install
time.dirinstall: once.install os.dirinstall io.dirinstall

