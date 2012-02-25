# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

units: yacc.go units.y
	go run yacc.go -p units_ units.y
	go build -o units y.go

clean:
	rm -f y.go y.output units
