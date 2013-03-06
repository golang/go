// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <windows.h>
#define NOPLAN9DEFINES
#include <libc.h>
#include "win.h"

int
runcmd(char **argv)
{
	// Mostly copied from ../cmd/dist/windows.c.
	// If there's a bug here, fix the logic there too.
	int i, j, nslash;
	Fmt fmt;
	char *q;
	WinRune *r;
	STARTUPINFOW si;
	PROCESS_INFORMATION pi;
	DWORD code;

	fmtstrinit(&fmt);
	for(i=0; argv[i]; i++) {
		if(i > 0)
			fmtprint(&fmt, " ");
		q = argv[i];
		if(strstr(q, " ") || strstr(q, "\t") || strstr(q, "\"") || strstr(q, "\\\\") || (strlen(q) > 0 && q[strlen(q)-1] == '\\')) {
			fmtprint(&fmt, "\"");
			nslash = 0;
			for(; *q; q++) {
				if(*q == '\\') {
					nslash++;
					continue;
				}
				if(*q == '"') {
					for(j=0; j<2*nslash+1; j++)
						fmtprint(&fmt, "\\");
					nslash = 0;
				}
				for(j=0; j<nslash; j++)
					fmtprint(&fmt, "\\");
				nslash = 0;
				fmtprint(&fmt, "\"");
			}
			for(j=0; j<2*nslash; j++)
				fmtprint(&fmt, "\\");
			fmtprint(&fmt, "\"");
		} else {
			fmtprint(&fmt, "%s", q);
		}
	}
	
	q = fmtstrflush(&fmt);
	r = torune(q);
	free(q);

	memset(&si, 0, sizeof si);
	si.cb = sizeof si;
	si.dwFlags = STARTF_USESTDHANDLES;
	si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
	si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

	if(!CreateProcessW(nil, r, nil, nil, TRUE, 0, nil, nil, &si, &pi)) {
		free(r);
		return -1;
	}

	free(r);
	if(WaitForMultipleObjects(1, &pi.hProcess, FALSE, INFINITE) != 0)
		return -1;
	i = GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	if(!i)
		return -1;
	if(code != 0) {
		werrstr("unsuccessful exit status: %d", (int)code);
		return -1;
	}
	return 0;
}
