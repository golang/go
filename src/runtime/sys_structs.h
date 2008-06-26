// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifdef amd64_linux
	#include "amd64_linux.h"
#else
	#ifdef amd64_darwin
		#include "amd64_darwin.h"
	#endif
#else
	You_need_to_write_the_syscall_header
#endif
