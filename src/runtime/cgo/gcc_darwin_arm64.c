// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <string.h> /* for strerror */
#include <sys/param.h>
#include <unistd.h>
#include <stdlib.h>

#include "libcgo.h"
#include "libcgo_unix.h"

#include <TargetConditionals.h>

#if TARGET_OS_IPHONE
#include <CoreFoundation/CFBundle.h>
#include <CoreFoundation/CFString.h>
#endif

static void *threadentry(void*);
static void (*setg_gcc)(void*);

void
_cgo_sys_thread_start(ThreadStart *ts)
{
	pthread_attr_t attr;
	sigset_t ign, oset;
	pthread_t p;
	size_t size;
	int err;

	//fprintf(stderr, "runtime/cgo: _cgo_sys_thread_start: fn=%p, g=%p\n", ts->fn, ts->g); // debug
	sigfillset(&ign);
	pthread_sigmask(SIG_SETMASK, &ign, &oset);

	size = pthread_get_stacksize_np(pthread_self());
	pthread_attr_init(&attr);
	pthread_attr_setstacksize(&attr, size);
	// Leave stacklo=0 and set stackhi=size; mstart will do the rest.
	ts->g->stackhi = size;
	err = _cgo_try_pthread_create(&p, &attr, threadentry, ts);

	pthread_sigmask(SIG_SETMASK, &oset, nil);

	if (err != 0) {
		fprintf(stderr, "runtime/cgo: pthread_create failed: %s\n", strerror(err));
		abort();
	}
}

extern void crosscall1(void (*fn)(void), void (*setg_gcc)(void*), void *g);
static void*
threadentry(void *v)
{
	ThreadStart ts;

	ts = *(ThreadStart*)v;
	free(v);

#if TARGET_OS_IPHONE
	darwin_arm_init_thread_exception_port();
#endif

	crosscall1(ts.fn, setg_gcc, (void*)ts.g);
	return nil;
}

#if TARGET_OS_IPHONE

// init_working_dir sets the current working directory to the app root.
// By default ios/arm64 processes start in "/".
static void
init_working_dir()
{
	CFBundleRef bundle = CFBundleGetMainBundle();
	if (bundle == NULL) {
		fprintf(stderr, "runtime/cgo: no main bundle\n");
		return;
	}
	CFURLRef url_ref = CFBundleCopyResourceURL(bundle, CFSTR("Info"), CFSTR("plist"), NULL);
	if (url_ref == NULL) {
		// No Info.plist found. It can happen on Corellium virtual devices.
		return;
	}
	CFStringRef url_str_ref = CFURLGetString(url_ref);
	char buf[MAXPATHLEN];
	Boolean res = CFStringGetCString(url_str_ref, buf, sizeof(buf), kCFStringEncodingUTF8);
	CFRelease(url_ref);
	if (!res) {
		fprintf(stderr, "runtime/cgo: cannot get URL string\n");
		return;
	}

	// url is of the form "file:///path/to/Info.plist".
	// strip it down to the working directory "/path/to".
	int url_len = strlen(buf);
	if (url_len < sizeof("file://")+sizeof("/Info.plist")) {
		fprintf(stderr, "runtime/cgo: bad URL: %s\n", buf);
		return;
	}
	buf[url_len-sizeof("/Info.plist")+1] = 0;
	char *dir = &buf[0] + sizeof("file://")-1;

	if (chdir(dir) != 0) {
		fprintf(stderr, "runtime/cgo: chdir(%s) failed\n", dir);
	}

	// The test harness in go_ios_exec passes the relative working directory
	// in the GoExecWrapperWorkingDirectory property of the app bundle.
	CFStringRef wd_ref = CFBundleGetValueForInfoDictionaryKey(bundle, CFSTR("GoExecWrapperWorkingDirectory"));
	if (wd_ref != NULL) {
		if (!CFStringGetCString(wd_ref, buf, sizeof(buf), kCFStringEncodingUTF8)) {
			fprintf(stderr, "runtime/cgo: cannot get GoExecWrapperWorkingDirectory string\n");
			return;
		}
		if (chdir(buf) != 0) {
			fprintf(stderr, "runtime/cgo: chdir(%s) failed\n", buf);
		}
	}
}

#endif // TARGET_OS_IPHONE

void
x_cgo_init(G *g, void (*setg)(void*))
{
	//fprintf(stderr, "x_cgo_init = %p\n", &x_cgo_init); // aid debugging in presence of ASLR
	setg_gcc = setg;
	_cgo_set_stacklo(g, NULL);

#if TARGET_OS_IPHONE
	darwin_arm_init_mach_exception_handler();
	darwin_arm_init_thread_exception_port();
	init_working_dir();
#endif
}
