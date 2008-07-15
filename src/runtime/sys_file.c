// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "sys_types.h"

void
sys·readfile(string filein, string fileout, bool okout)
{
	int32 fd;
	byte namebuf[256];
	struct stat statbuf;

	fileout = nil;
	okout = false;

	if(filein == nil || filein->len >= sizeof(namebuf))
		goto out;

	mcpy(namebuf, filein->str, filein->len);
	namebuf[filein->len] = '\0';
	fd = open(namebuf, 0);
	if(fd < 0)
		goto out;

	if (fstat(fd, &statbuf) < 0)
		goto close_out;

	if (statbuf.st_size <= 0)
		goto close_out;

	fileout = mal(sizeof(fileout->len)+statbuf.st_size + 1);
	fileout->len = statbuf.st_size;

	if (read(fd, fileout->str, statbuf.st_size) != statbuf.st_size) {
		fileout = nil;
		goto close_out;
	}
	okout = true;

close_out:
	close(fd);
out:
	FLUSH(&fileout);
	FLUSH(&okout);
}

void
sys·writefile(string filein, string textin, bool okout)
{
	int32 fd;
	byte namebuf[256];

	okout = false;

	if(filein == nil || filein->len >= sizeof(namebuf))
		goto out;

	mcpy(namebuf, filein->str, filein->len);
	namebuf[filein->len] = '\0';
	fd = open(namebuf, 1|O_CREAT, 0644);  // open for write, create if non-existant (sic)
	if(fd < 0)
		goto out;

	if (write(fd, textin->str, textin->len) != textin->len) {
		goto close_out;
	}
	okout = true;

close_out:
	close(fd);
out:
	FLUSH(&okout);
}
