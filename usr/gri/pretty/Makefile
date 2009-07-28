# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.conf
include $(GOROOT)/src/Make.$(GOARCH)

all: godoc gofmt

godoc: godoc.$O
	$(LD) -o godoc godoc.$O

gofmt: gofmt.$O
	$(LD) -o gofmt gofmt.$O

test: gofmt
	./test.sh

smoketest: gofmt
	./test.sh $(GOROOT)/src/pkg/go/printer/printer.go

install: pretty godoc untab
	cp godoc $(GOBIN)/godoc
	cp gofmt $(GOBIN)/gofmt

clean:
	rm -f godoc gofmt *.$O *.a 6.out *~

%.$O:	%.go
	$(GC) $(F) $<
