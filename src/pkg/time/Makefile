# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

genzabbrs: genzabbrs.go
	go build genzabbrs.go

windows: genzabbrs
	./genzabbrs | gofmt >zoneinfo_abbrs_windows.go
