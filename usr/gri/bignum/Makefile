# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

G=6g
L=6l

bignum_test: bignum_test.6
	$(L) -o bignum_test bignum_test.6

test: bignum_test
	./bignum_test

clean:
	rm -f bignum_test *.6  *~

bignum_test.6:	bignum.6

%.6:	%.go
	$(G) $(F) $<
