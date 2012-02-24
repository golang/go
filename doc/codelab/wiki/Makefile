# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

all: index.html

CLEANFILES:=srcextract.bin htmlify.bin get.bin

index.html: wiki.html srcextract.bin htmlify.bin
	PATH=.:$$PATH awk '/^!/{system(substr($$0,2)); next} {print}' < wiki.html | tr -d '\r' > index.html

test: get.bin
	bash ./test.sh
	rm -f get.6 get.bin

%.bin: %.go
	go build -o $@ $^

clean:
	rm -f $(CLEANFILES)
