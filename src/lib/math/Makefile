# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CFLAGS=
O=6
CC=$(O)c
AS=$(O)a
GC=$(O)g

# TODO(r): building directly in the target makes internal dependencies self-consistent.
# need to address this a better way.
PKG=$(GOROOT)/pkg/math.a

O1=\
	atan.$O fabs.$O floor.$O fmod.$O hypot.$O log.$O pow10.$O sin.$O sqrt.$O tan.$O
O2=\
	asin.$O atan2.$O exp.$O
O3=\
	pow.$O sinh.$O
O4=\
	tanh.$O

install: $(PKG)

$(PKG): a1 a2 a3 a4

a1: $(O1)
	$(O)ar grc $(PKG) $(O1)

a2: $(O2)
	$(O)ar grc $(PKG) $(O2)

a3: $(O3)
	$(O)ar grc $(PKG) $(O3)

a4: $(O4)
	$(O)ar grc $(PKG) $(O4)

$(O1): nuke
$(O2): a1
$(O3): a2
$(O4): a3

nuke:
	rm -f *.$(O) *.a $(PKG)

clean:
	rm -f *.$(O) *.a

%.$O:	%.go
	$(GC) $<
