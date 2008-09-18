# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=
O=6
CC=$(O)c
AS=$(O)a
GC=$(O)g

PKG=$(GOROOT)/pkg/syscall.a

O1=\
	syscall.$O \
	errstr_$(GOOS).$O \

O2=\
	file_$(GOARCH)_$(GOOS).$O \
	time_$(GOARCH)_$(GOOS).$O \
	syscall_$(GOARCH)_$(GOOS).$O \


install: nuke $(PKG)

$(PKG): a1 a2

a1: 	$(O1)
	$(O)ar grc $(PKG) $(O1)
	rm *.6

a2: 	$(O2)
	$(O)ar grc $(PKG) $(O2)
	rm *.6

$(O1): nuke
$(O2): a1

nuke:
	rm -f *.$(O) *.a $(PKG)

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
