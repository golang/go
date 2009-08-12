# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=syscall
GOFILES=\
	errstr.go\
	exec.go\
	syscall.go\
	syscall_$(GOOS).go\
	syscall_$(GOOS)_$(GOARCH).go\
	zerrors_$(GOOS)_$(GOARCH).go\
	zsyscall_$(GOOS)_$(GOARCH).go\
	zsysnum_$(GOOS)_$(GOARCH).go\
	ztypes_$(GOOS)_$(GOARCH).go\

OFILES=\
	asm_$(GOOS)_$(GOARCH).$O\
	types_$(GOOS)_$(GOARCH).$O\

include $(GOROOT)/src/Make.pkg
