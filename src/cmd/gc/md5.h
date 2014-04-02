// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

typedef struct MD5 MD5;
struct MD5
{
	uint32 s[4];
	uchar x[64];
	int nx;
	uint64 len;
};

void md5reset(MD5*);
void md5write(MD5*, uchar*, int);
uint64 md5sum(MD5*, uint64*);
