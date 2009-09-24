# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=cgo
GOFILES=\
	ast.go\
	gcc.go\
	main.go\
	out.go\
	util.go\

include $(GOROOT)/src/Make.cmd

# Tests
# TODO(rsc): Delete

gmp:
	make cgo
	cgo gmp.go
	gcc -fPIC -O2 -o gcc.o -c _cgo_gcc.c
	gcc -shared -o gmp.so gcc.o -L$(GOROOT)/pkg/$(GOOS)_$(GOARCH) -lcgo -lgmp
	6c -D_64BIT -I$(GOROOT)/src/pkg/runtime _cgo_c.c
	6g _cgo_go.go
	gopack grc gmp.a _cgo_c.6 _cgo_go.6
	6g pidigits.go
	6l pidigits.6
	LD_LIBRARY_PATH=.:$(GOROOT)/pkg/$(GOOS)_$(GOARCH) 6.out

stdio:
	make cgo
	cgo stdio.go
	gcc -fPIC -O2 -o gcc.o -c _cgo_gcc.c
	gcc -shared -Wl,--rpath -Wl,$(GOROOT)/pkg/$(GOOS)_$(GOARCH) -o main.so gcc.o -L$(GOROOT)/pkg/$(GOOS)_$(GOARCH) -lcgo
	6c -D_64BIT -I$(GOROOT)/src/pkg/runtime _cgo_c.c
	6g _cgo_go.go
	6l _cgo_c.6 _cgo_go.6
	LD_LIBRARY_PATH=.:$(GOROOT)/pkg/$(GOOS)_$(GOARCH) 6.out
