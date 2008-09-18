# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

O=6
GC=$(O)g
AS=$(O)a

NET=$(GOROOT)/pkg/net.a
SOCKET=$(GOROOT)/pkg/socket.a
IP=$(GOROOT)/pkg/ip.$O

NETO=\
	net.$O\

SOCKETO=\
	cvt.$O\
	socket_$(GOOS).$O\

$(NET): $(NETO)
	$(O)ar grc $(NET) $(NETO)

$(NETO): $(IP) $(SOCKET)

$(SOCKETO): $(IP)

$(SOCKET): $(SOCKETO)
	$(O)ar grc $(SOCKET) $(SOCKETO)
	rm $(SOCKETO)

$(GOROOT)/pkg/%.$O: %.$O
	cp $*.$O $(GOROOT)/pkg/$*.$O
	rm $*.$O

install: nuke $(IP) $(SOCKET) $(NET)

nuke:
	rm -f *.$O *.a $(IP) $(NET)

clean:
	rm -f *.$O *.a

%.$O:	%.go
	$(GC) $<

%.$O:	%.s
	$(AS) $<
