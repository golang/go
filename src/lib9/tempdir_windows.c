// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <windows.h>
#include <libc.h>
#include "win.h"

char*
toutf(WinRune *r)
{
	Rune *r1;
	int i, n;
	char *p;
	
	n = 0;
	while(r[n] != '\0')
		n++;
	n++;
	r1 = malloc(n*sizeof r1[0]);
	for(i=0; i<n; i++)
		r1[i] = r[i];
	p = smprint("%S", r1);
	free(r1);
	return p;
}

WinRune*
torune(char *p)
{
	int i, n;
	Rune rr;
	WinRune *r;
	
	n = utflen(p);
	r = malloc((n+1)*sizeof r[0]);
	for(i=0; i<n; i++) {
		p += chartorune(&rr, p);
		r[i] = rr;
	}
	r[n] = '\0';
	return r;
}

char*
mktempdir(void)
{
	WinRune buf[1024];
	WinRune tmp[MAX_PATH];
	WinRune golink[] = {'g', 'o', 'l', 'i', 'n', 'k', '\0'};
	int n;
	
	n = GetTempPathW(nelem(buf), buf);
	if(n <= 0)
		return nil;
	buf[n] = '\0';
	
	if(GetTempFileNameW(buf, golink, 0, tmp) == 0)
		return nil;
	DeleteFileW(tmp);
	if(!CreateDirectoryW(tmp, nil))
		return nil;
	
	return toutf(tmp);
}

void
removeall(char *p)
{
	WinRune *r, *r1;
	DWORD attr;
	char *q, *qt, *elem;
	HANDLE h;
	WIN32_FIND_DATAW data;
	
	r = torune(p);
	attr = GetFileAttributesW(r);
	if(attr == INVALID_FILE_ATTRIBUTES || !(attr & FILE_ATTRIBUTE_DIRECTORY)) {
		DeleteFileW(r);
		free(r);
		return;
	}

	q = smprint("%s\\*", p);
	r1 = torune(q);
	free(q);
	h = FindFirstFileW(r1, &data);
	if(h == INVALID_HANDLE_VALUE)
		goto done;
	do{
		q = toutf(data.cFileName);
		elem = strrchr(q, '\\');
		if(elem != nil)
			elem++;
		else
			elem = q;
		if(strcmp(elem, ".") == 0 || strcmp(elem, "..") == 0) {
			free(q);
			continue;
		}
		qt = smprint("%s\\%s", p, q);
		free(q);
		removeall(qt);
		free(qt);
	}while(FindNextFileW(h, &data));
	FindClose(h);

done:
	free(r1);
	RemoveDirectoryW(r);
	free(r);
}
