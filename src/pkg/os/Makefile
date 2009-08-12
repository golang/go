# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include $(GOROOT)/src/Make.$(GOARCH)

TARG=os
GOFILES=\
	dir_$(GOOS)_$(GOARCH).go\
	env.go\
	error.go\
	exec.go\
	file.go\
	getwd.go\
	path.go\
	proc.go\
	stat_$(GOOS)_$(GOARCH).go\
	sys_$(GOOS).go\
	time.go\
	types.go\

include $(GOROOT)/src/Make.pkg
