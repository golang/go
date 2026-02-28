// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <string.h>

#include "_cgo_export.h"

void
callback(void *f)
{
	// use some stack space
	volatile char data[64*1024];

	data[0] = 0;
	goCallback(f);
        data[sizeof(data)-1] = 0;
}

void
callGoFoo(void)
{
	extern void goFoo(void);
	goFoo();
}

void
IntoC(void)
{
	BackIntoGo();
}

void
Issue1560InC(void)
{
	Issue1560FromC();
}

void
callGoStackCheck(void)
{
	extern void goStackCheck(void);
	goStackCheck();
}

int
returnAfterGrow(void)
{
	extern int goReturnVal(void);
	goReturnVal();
	return 123456;
}

int
returnAfterGrowFromGo(void)
{
	extern int goReturnVal(void);
	return goReturnVal();
}

void
callGoWithString(void)
{
	extern void goWithString(GoString);
	const char *str = "string passed from C to Go";
	goWithString((GoString){str, strlen(str)});
}
