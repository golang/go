// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os_GOOS.h"

int64
runtime·nanotime(void)
{
	static int32 fd = -1;
	byte b[8];
	uint32 hi, lo;

	// As long as all goroutines share the same file
	// descriptor table we can get away with using
	// just a static fd.  Without a lock the file can
	// be opened twice but that's okay.
	//
	// Using /dev/bintime gives us a latency on the
	// order of ten microseconds between two calls.
	//
	// The naïve implementation (without the cached
	// file descriptor) is roughly four times slower
	// in 9vx on a 2.16 GHz Intel Core 2 Duo.

	if(fd < 0 && (fd = runtime·open("/dev/bintime", OREAD|OCEXEC, 0)) < 0)
		return 0;
	if(runtime·pread(fd, b, sizeof b, 0) != sizeof b)
		return 0;
	hi = b[0]<<24 | b[1]<<16 | b[2]<<8 | b[3];
	lo = b[4]<<24 | b[5]<<16 | b[6]<<8 | b[7];
	return (int64)hi<<32 | (int64)lo;
}
