# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=syscall
GOFILES=\
	str.go\
	exec.go\
	syscall.go\
	syscall_$(GOARCH).go\
	syscall_$(GOOS).go\
	syscall_$(GOOS)_$(GOARCH).go\
	zerrors_$(GOOS)_$(GOARCH).go\
	zsyscall_$(GOOS)_$(GOARCH).go\
	zsysnum_$(GOOS)_$(GOARCH).go\
	ztypes_$(GOOS)_$(GOARCH).go\

GOFILES_freebsd=\
	syscall_unix.go\

GOFILES_darwin=\
	syscall_unix.go\

GOFILES_linux=\
	syscall_unix.go\

GOFILES_nacl=\
	syscall_unix.go\

OFILES=\
	asm_$(GOOS)_$(GOARCH).$O\

GOFILES+=$(GOFILES_$(GOOS))

include ../../Make.pkg
