# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=
O=6
CC=$(O)c
AS=$(O)a
GC=$(O)g

PKG=syscall.a

OFILES=\
	syscall.$O \
	errstr_$(GOOS).$O \
	file_$(GOARCH)_$(GOOS).$O \
	syscall_$(GOARCH)_$(GOOS).$O \


install: $(PKG)
	cp $(PKG) $(GOROOT)/pkg

$(PKG): $(OFILES)
	$(O)ar grc $(PKG) $(OFILES)

nuke:
	rm -f *.$(O) *.a $(GOROOT)/lib/$(PKG)

clean:
	rm -f *.$(O) *.a

%.$O:	%.c
	$(CC) $<

sys_file.$O:	sys_file.c sys_types.h $(OS_H)
	$(CC) -D$(GOARCH)_$(GOOS) $<

%.$O:	%.s
	$(AS) $<

%.$O:	%.go
	$(GC) $<
