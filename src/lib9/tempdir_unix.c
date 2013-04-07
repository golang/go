// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

#include <u.h>
#include <dirent.h>
#include <sys/stat.h>
#define NOPLAN9DEFINES
#include <libc.h>

char*
mktempdir(void)
{
	char *tmp, *p;
	
	tmp = getenv("TMPDIR");
	if(tmp == nil || strlen(tmp) == 0)
		tmp = "/var/tmp";
	p = smprint("%s/go-link-XXXXXX", tmp);
	if(mkdtemp(p) == nil)
		return nil;
	return p;
}

void
removeall(char *p)
{
	DIR *d;
	struct dirent *dp;
	char *q;
	struct stat st;

	if(stat(p, &st) < 0)
		return;
	if(!S_ISDIR(st.st_mode)) {
		unlink(p);
		return;
	}

	d = opendir(p);
	while((dp = readdir(d)) != nil) {
		if(strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0)
			continue;
		q = smprint("%s/%s", p, dp->d_name);
		removeall(q);
		free(q);
	}
	closedir(d);
	rmdir(p);
}
