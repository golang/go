# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=
O=6
CC=$(O)c
AS=$(O)a

LIB=lib_$(GOARCH)_$(GOOS).a

RT0OFILES=\
	rt0_$(GOARCH)_$(GOOS).$O\

LIBOFILES=\
	rt0_$(GOARCH).$O\
	rt1_$(GOARCH)_$(GOOS).$O\
	rt2_$(GOARCH).$O\
	sys_$(GOARCH)_$(GOOS).$O\
	runtime.$O\
	hashmap.$O\
	chan.$O\
	iface.$O\
	array.$O\
	print.$O\
	rune.$O\
	proc.$O\
	string.$O\
	sys_file.$O\

OFILES=$(RT0OFILES) $(LIBOFILES)
OS_H=$(GOARCH)_$(GOOS).h
HFILES=runtime.h hashmap.h $(OS_H_)

install: rt0 $(LIB) runtime.acid
	cp $(RT0OFILES) $(GOROOT)/lib
	cp $(LIB) $(GOROOT)/lib
	cp runtime.acid $(GOROOT)/acid/runtime.acid

rt0:	$(RT0OFILES)

$(LIB): $(LIBOFILES)
	$(O)ar rc $(LIB) $(LIBOFILES)

$(OFILES): $(HFILES)

nuke:
	rm -f *.$(O) *.a $(GOROOT)/lib/$(LIB)

clean:
	rm -f *.$(O) *.a runtime.acid

%.$O:	%.c
	$(CC) -w $<

sys_file.$O:	sys_file.c sys_types.h $(OS_H)
	$(CC) -w -D$(GOARCH)_$(GOOS) $<

%.$O:	%.s
	$(AS) $<

runtime.acid: runtime.h proc.c
	$(CC) -a proc.c >runtime.acid
