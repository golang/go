// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file. 

#include "runtime.h"

extern void runtime·write(int32 fd, void *v, int32 len, int32 cap);	// slice, spelled out

int32
runtime·write(int32 fd, void *v, int32 len)
{
	runtime·write(fd, v, len, len);
	return len;
}

void
runtime·gettime(int64*, int32*) 
{
}
