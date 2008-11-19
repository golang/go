# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.conf

TARG=gotest

clean:
	@true

install: $(TARG)
	test -f $(BIN)/$(TARG) && chmod u+w $(BIN)/$(TARG)
	cp $(TARG) $(BIN)/$(TARG)
