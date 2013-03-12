// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os_GOOS.h"

byte*
runtime·getenv(int8 *s)
{
	int32 fd, len, n, r;
	byte file[128];
	byte *p;

	len = runtime·findnull((byte*)s);
	if(len > sizeof file-6)
		return nil;

	runtime·memclr(file, sizeof file);
	runtime·memmove((void*)file, (void*)"/env/", 5);
	runtime·memmove((void*)(file+5), (void*)s, len);

	fd = runtime·open((int8*)file, OREAD, 0);
	if(fd < 0)
		return nil;
	n = runtime·seek(fd, 0, 2);
	p = runtime·malloc(n+1);
	r = runtime·pread(fd, p, n, 0);
	runtime·close(fd);
	if(r < 0)
		return nil;
	return p;
}
