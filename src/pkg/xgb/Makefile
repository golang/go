# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.$(GOARCH)

TARG=xgb
GOFILES=\
	auth.go\
	xgb.go\
	xproto.go\

include ../../Make.pkg

xproto: proto/src/xproto.xml
	python go_client.py -p proto/ proto/src/xproto.xml
	gofmt -w xproto.go

proto/src/xproto.xml:
	git clone git://anongit.freedesktop.org/git/xcb/proto

