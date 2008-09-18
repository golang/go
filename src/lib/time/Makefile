# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

O=6
GC=$(O)g

PKG=$(GOROOT)/pkg/time.a

O1=\
	zoneinfo.$O
O2=\
	time.$O\

install: nuke $(PKG)

$(PKG): a1 a2

a1: 	$(O1)
	$(O)ar grc $(PKG) $(O1)

a2: 	$(O2)
	$(O)ar grc $(PKG) $(O2)

$(O1): nuke
$(O2): a1

nuke:
	rm -f *.$(O) *.a $(PKG)

clean:
	rm -f *.$(O) *.a

%.$O:	%.go
	$(GC) $<
