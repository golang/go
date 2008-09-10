# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

O=6
GC=$(O)g

PKG=os.a

OFILES=\
	os.$O \
	os_file.$O \


install: $(PKG)
	cp $(PKG) $(GOROOT)/pkg/$(PKG)

$(PKG): $(OFILES)
	$(O)ar grc $(PKG) $(OFILES)

nuke:
	rm -f *.$(O) *.a $(GOROOT)/pkg/$(PKG)

clean:
	rm -f *.$(O) *.a

%.$O:	%.go
	$(GC) $<
