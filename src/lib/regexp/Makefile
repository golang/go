# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

A=6
G=$(A)g
L=$(A)l

all: main

main:	main.6
	$L -o main main.6

main.6: regexp.6

clean:
	rm -f *.6 main

%.6:	%.go
	$G $<
