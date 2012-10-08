# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

chars:
	go run ../maketables.go -tables=chars -package=main > chars.go
	gofmt -w -s chars.go
