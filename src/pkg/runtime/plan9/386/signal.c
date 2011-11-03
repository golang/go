// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file. 

#include "runtime.h"

int64
runtime·nanotime(void) 
{
	// Won't compile.
}

String
runtime·signame(int32)
{
	return runtime·emptystring;
}

void
runtime·resetcpuprofiler(int32 hz)
{
	// TODO: Enable profiling interrupts.
	
	m->profilehz = hz;
}
