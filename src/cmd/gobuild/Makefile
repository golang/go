# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# sadly, not auto-generated

O=6
OS=568vqo
GC=$(O)g
CC=$(O)c -FVw
AS=$(O)a
AR=6ar
LD=$(O)l

default: gobuild

clean:
	rm -f *.[$(OS)] *.a [$(OS)].out gobuild

%.$O: %.go
	$(GC) $*.go

O1=util.$O
O2=makefile.$O
O3=gobuild.$O

phases: a1 a2 a3
gobuild.a: phases

a1: $(O1)
	$(AR) grc gobuild.a util.$O
	rm -f $(O1)

a2: $(O2)
	$(AR) grc gobuild.a makefile.$O
	rm -f $(O2)

a3: $(O3)
	$(AR) grc gobuild.a gobuild.$O
	rm -f $(O3)

newpkg: clean
	$(AR) grc gobuild.a

$(O1): newpkg
$(O2): a1
$(O3): a2

# zzgobuild is a fake target that will always run, even if
# "gobuild" existed at the beginning of the make.
# The problem is that if you "make install" and install
# depends on gobuild and this rule says gobuild,
# and gobuild.a depends on phases, "phases" gets
# run, which cleans everything and then rebuilds
# gobuild.a.  So now make thinks gobuild was up to date
# to begin with (and it ran "phases" just for good measure)
# but in fact gobuild is gone ("phases" removed it).
#
# Calling the target zzgobuild instead means that
# make will always run this rule, rebuilding gobuild
# before trying to install it.  Sigh.
zzgobuild: main.$O gobuild.a
	$(LD) -o gobuild main.$O

gobuild: zzgobuild

main.$O: phases

nuke: clean
	rm -f $(HOME)/bin/gobuild

install: zzgobuild
	cp gobuild $(HOME)/bin/gobuild

