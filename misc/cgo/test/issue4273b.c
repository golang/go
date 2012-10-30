// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifdef __ELF__
extern void _compilerrt_abort_impl(const char *file, int line, const char *func);

void __my_abort(const char *file, int line, const char *func) {
	_compilerrt_abort_impl(file, line, func);
}
#endif
