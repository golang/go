# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

TARG=expr$(shell go env GOEXE)

$(TARG): yacc.go expr.y
	go run yacc.go -p expr expr.y
	go build -o $(TARG) y.go

clean:
	rm -f y.go y.output $(TARG)
