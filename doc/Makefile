# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

RAWHTML=\
	articles/defer_panic_recover.rawhtml\
	articles/error_handling.rawhtml\
	articles/slices_usage_and_internals.rawhtml\
	articles/laws_of_reflection.rawhtml\
	articles/c_go_cgo.rawhtml\
	articles/concurrency_patterns.rawhtml\
	articles/godoc_documenting_go_code.rawhtml\
	articles/gobs_of_data.rawhtml\
	articles/json_and_go.rawhtml\
	articles/json_rpc_tale_of_interfaces.rawhtml\
	articles/image_draw.rawhtml\
	articles/image_package.rawhtml\
	effective_go.rawhtml\
	go1.rawhtml\

all: $(RAWHTML)

%.rawhtml: %.html
	godoc -url /doc/$< >$@

clean:
	rm -f $(RAWHTML)

compare:
	for i in $(RAWHTML); do \
		godoc -url /doc/$${i/.rawhtml/.html} | diff -u $$i -; \
	done
