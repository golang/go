# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=math

OFILES_amd64=\
	sqrt_amd64.$O\

OFILES=\
	$(OFILES_$(GOARCH))

ALLGOFILES=\
	asin.go\
	atan.go\
	atan2.go\
	const.go\
	exp.go\
	fabs.go\
	floor.go\
	fmod.go\
	hypot.go\
	log.go\
	pow.go\
	pow10.go\
	runtime.go\
	sin.go\
	sinh.go\
	sqrt.go\
	tan.go\
	tanh.go\

NOGOFILES=\
	$(subst _$(GOARCH).$O,.go,$(OFILES_$(GOARCH)))

GOFILES=\
	$(filter-out $(NOGOFILES),$(ALLGOFILES))\
	$(subst .go,_decl.go,$(NOGOFILES))\

include $(GOROOT)/src/Make.pkg
