// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file. 

#include "runtime.h"

void
runtime·gettime(int64*, int32*) 
{
}

String
runtime·signame(int32)
{
	return runtime·emptystring;
}
