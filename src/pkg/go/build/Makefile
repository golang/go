# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

syslist.go: ../../../Make.inc Makefile
	echo '// Generated automatically by make.' >$@
	echo 'package build' >>$@
	echo >>$@
	echo 'const goosList = "$(GOOS_LIST)"' >>$@
	echo 'const goarchList = "$(GOARCH_LIST)"' >>$@
