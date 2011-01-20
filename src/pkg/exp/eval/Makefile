# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../../Make.inc

TARG=exp/eval
GOFILES=\
	abort.go\
	bridge.go\
	compiler.go\
	expr.go\
	expr1.go\
	func.go\
	scope.go\
	stmt.go\
	type.go\
	typec.go\
	value.go\
	world.go\

include ../../../Make.pkg

main.$O: main.go $(pkgdir)/$(TARG).a
	$(GC) $<

eval: main.$O
	$(LD) -o $@ $<

gen.$O: gen.go
	$(GC) $<

generate: gen.$O
	$(LD) -o $@ $<;\
	./generate > expr1.go;\
	gofmt -w expr1.go

