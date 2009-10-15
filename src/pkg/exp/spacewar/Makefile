# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: 8.out

pdp1.8: pdp1.go
	8g pdp1.go

spacewar.8: spacewar.go code.go pdp1.8
	8g spacewar.go code.go

8.out: spacewar.8
	8l spacewar.8

clean:
	rm -f *.8 8.out

