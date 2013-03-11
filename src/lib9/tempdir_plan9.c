// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>

char*
mktempdir(void)
{
	char *p;
	int fd, i;
	
	p = smprint("/tmp/go-link-XXXXXX");
	for(i=0; i<1000; i++) {
		sprint(p, "/tmp/go-link-%06x", nrand((1<<24)-1));
		fd = create(p, OREAD|OEXCL, 0700|DMDIR);
		if(fd >= 0) {
			close(fd);
			return p;
		}
	}
	free(p);
	return nil;
}

void
removeall(char *p)
{
	int fd, n, i;
	Dir *d;
	char *q;
	
	if(remove(p) >= 0)
		return;
	if((d = dirstat(p)) == nil)
		return;
	if(!(d->mode & DMDIR)) {
		free(d);
		return;
	}
	free(d);
	
	if((fd = open(p, OREAD)) < 0)
		return;
	n = dirreadall(fd, &d);
	close(fd);
	for(i=0; i<n; i++) {
		q = smprint("%s/%s", p, d[i].name);
		removeall(q);
		free(q);
	}
	free(d);
}
