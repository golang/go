# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

RAWHTML=\
	articles/defer_panic_recover.rawhtml\
	articles/error_handling.rawhtml\
	articles/slices_usage_and_internals.rawhtml\
	articles/laws_of_reflection.rawhtml\
	effective_go.rawhtml\
	go1.rawhtml\

all: $(RAWHTML)

%.rawhtml: %.html
	godoc -url /doc/$* >$@

clean:
	rm -f $(RAWHTML) 
