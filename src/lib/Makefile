# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: install

GC=6g

DIRS=\
	bignum\
	bufio\
	container\
	exec\
	exvar\
	flag\
	fmt\
	go\
	hash\
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
	bignum \
	bufio \
	container \
	exec \
	exvar \
	flag \
	fmt \
	go \
	hash \
	http \
	io \
	json \
	log \
	math \
	net \
	once \
	os \
	path \
	reflect \
	regexp \
	sort \
	strconv \
	strings \
	sync \
	tabwriter \
	template \
	time \
	unicode \
	utf8 \

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
log.install: fmt.install io.install os.install time.install
path.install: io.install
once.install: sync.install
strings.install: utf8.install
testing.install: flag.install fmt.install

fmt.install: io.install reflect.install strconv.install
go.install: regexp.install sort.install strconv.install strings.install \
	utf8.install unicode.install fmt.install
hash.install: os.install
http.install: bufio.install io.install net.install os.install path.install strings.install log.install
io.install: os.install sync.install
json.install: container.install fmt.install io.install math.install \
	strconv.install strings.install utf8.install
# TODO(rsc): net is not supposed to depend on fmt or strings or strconv
net.install: fmt.install once.install os.install strconv.install strings.install
os.install: syscall.install once.install
regexp.install: os.install
reflect.install: strconv.install sync.install
strconv.install: math.install os.install utf8.install
sync.install:
syscall.install: sync.install
tabwriter.install: os.install io.install container.install
template.install: fmt.install io.install os.install reflect.install strings.install
time.install: once.install os.install io.install

