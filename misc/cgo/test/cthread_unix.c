// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

#include <pthread.h>
#include "_cgo_export.h"

static void*
addThread(void *p)
{
	int i, max;
	
	max = *(int*)p;
	for(i=0; i<max; i++)
		Add(i);
	return 0;
}

void
doAdd(int max, int nthread)
{
	enum { MaxThread = 20 };
	int i;
	pthread_t thread_id[MaxThread];
	
	if(nthread > MaxThread)
		nthread = MaxThread;
	for(i=0; i<nthread; i++)
		pthread_create(&thread_id[i], 0, addThread, &max);
	for(i=0; i<nthread; i++)
		pthread_join(thread_id[i], 0);		
}
