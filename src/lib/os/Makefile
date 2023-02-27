# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

O=6
GC=$(O)g

PKG=$(GOROOT)/pkg/os.a

O1=\
	os.$O

install: $(PKG)

$(PKG): a1

a1: 	$(O1)
	$(O)ar grc $(PKG) $(O1)

nuke:
	rm -f *.$(O) *.a $(PKG)

clean:
	rm -f *.$(O) *.a

%.$O:	%.go
	$(GC) $<
