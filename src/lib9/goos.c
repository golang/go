// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>

static char*
defgetenv(char *name, char *def)
{
	char *p;
	
	p = getenv(name);
	if(p == nil || p[0] == '\0')
		p = def;
	return p;
}

char*
getgoos(void)
{
	return defgetenv("GOOS", GOOS);
}

char*
getgoarch(void)
{
	return defgetenv("GOARCH", GOARCH);
}

char*
getgoroot(void)
{
	return defgetenv("GOROOT", GOROOT);
}

char*
getgoversion(void)
{
	return GOVERSION;
}

char*
getgoarm(void)
{
	return defgetenv("GOARM", GOARM);
}

char*
getgo386(void)
{
	return defgetenv("GO386", GO386);
}

char *
getgoextlinkenabled(void)
{
	return GO_EXTLINK_ENABLED;
}
