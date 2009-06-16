# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=gofmt
OFILES=\
	gofmt.$O\

$(TARG): $(OFILES)
	$(LD) -o $(TARG) $(OFILES)

test: $(TARG)
	./test.sh

smoketest: $(TARG)
	./test.sh $(GOROOT)/src/pkg/go/parser/parser.go

clean:
	rm -f $(OFILES) $(TARG)

install: $(TARG)
	cp $(TARG) $(HOME)/bin/$(TARG)

%.$O:	%.go
	$(GC) $<
