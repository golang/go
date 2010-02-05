# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=math

OFILES_amd64=\
	sqrt_amd64.$O\

OFILES_386=\
	asin_386.$O\
	atan_386.$O\
	atan2_386.$O\
	exp_386.$O\
	fabs_386.$O\
	floor_386.$O\
	fmod_386.$O\
	hypot_386.$O\
	log_386.$O\
	modf_386.$O\
	sin_386.$O\
	sqrt_386.$O\
	tan_386.$O\

OFILES=\
	$(OFILES_$(GOARCH))

ALLGOFILES=\
	acosh.go\
	asin.go\
	asinh.go\
	atan.go\
	atanh.go\
	atan2.go\
	bits.go\
	const.go\
	copysign.go\
	erf.go\
	exp.go\
	expm1.go\
	fabs.go\
	floor.go\
	fmod.go\
	hypot.go\
	log.go\
	log1p.go\
	modf.go\
	pow.go\
	pow10.go\
	sin.go\
	sinh.go\
	sqrt.go\
	sqrt_port.go\
	tan.go\
	tanh.go\
	unsafe.go\

NOGOFILES=\
	$(subst _$(GOARCH).$O,.go,$(OFILES_$(GOARCH)))

GOFILES=\
	$(filter-out $(NOGOFILES),$(ALLGOFILES))\
	$(subst .go,_decl.go,$(NOGOFILES))\

include ../../Make.pkg
