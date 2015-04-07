// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

#ifdef WIN32
// A Windows DLL is unable to call an arbitrary function in
// the main executable. Work around that by making the main
// executable pass the callback function pointer to us.
void (*goCallback)(void);
__declspec(dllexport) void setCallback(void *f)
{
	goCallback = (void (*)())f;
}
__declspec(dllexport) void sofunc(void);
#else
extern void goCallback(void);
void setCallback(void *f) { (void)f; }
#endif

// OpenBSD and older Darwin lack TLS support
#if !defined(__OpenBSD__) && !defined(__APPLE__)
__thread int tlsvar = 12345;
#endif

void sofunc(void)
{
	goCallback();
}
