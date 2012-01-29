# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include ../../Make.inc
O:=$(HOST_O)

# The directory is nm because the source is portable and general.
# We call the binary 6nm to avoid confusion with the host nm.

TARG=nm
OFILES=\
	nm.$O\

include ../../Make.ctool
