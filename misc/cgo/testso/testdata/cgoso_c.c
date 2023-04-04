// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

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
#elif defined(_AIX)
// AIX doesn't allow the creation of a shared object with an
// undefined symbol. It's possible to bypass this problem by
// using -Wl,-G and -Wl,-brtl option which allows run-time linking.
// However, that's not how most of AIX shared object works.
// Therefore, it's better to consider goCallback as a pointer and
// to set up during an init function.
void (*goCallback)(void);
void setCallback(void *f) { goCallback = f; }
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
