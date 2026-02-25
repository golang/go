// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <limits.h>
#include <string.h> /* for strerror */
#include <sys/param.h>
#include <unistd.h>
#include <stdlib.h>

#include "libcgo.h"

#include <TargetConditionals.h>

#if TARGET_OS_IPHONE
#include <CoreFoundation/CFBundle.h>
#include <CoreFoundation/CFString.h>
#endif

#if TARGET_OS_IPHONE

static void
threadentry_platform(void)
{
#if TARGET_OS_IPHONE
	darwin_arm_init_thread_exception_port();
#endif
}

// init_working_dir sets the current working directory to the app root.
// By default ios/arm64 processes start in "/".
static void
init_working_dir()
{
	CFBundleRef bundle;
	CFURLRef url_ref;
	CFStringRef url_str_ref;
	char buf[MAXPATHLEN];
	Boolean res;
	int url_len;
	char *dir;
	CFStringRef wd_ref;

	bundle = CFBundleGetMainBundle();
	if (bundle == NULL) {
		fprintf(stderr, "runtime/cgo: no main bundle\n");
		return;
	}
	url_ref = CFBundleCopyResourceURL(bundle, CFSTR("Info"), CFSTR("plist"), NULL);
	if (url_ref == NULL) {
		// No Info.plist found. It can happen on Corellium virtual devices.
		return;
	}
	url_str_ref = CFURLGetString(url_ref);
	res = CFStringGetCString(url_str_ref, buf, sizeof(buf), kCFStringEncodingUTF8);
	CFRelease(url_ref);
	if (!res) {
		fprintf(stderr, "runtime/cgo: cannot get URL string\n");
		return;
	}

	// url is of the form "file:///path/to/Info.plist".
	// strip it down to the working directory "/path/to".
	url_len = strlen(buf);
	if (url_len < sizeof("file://")+sizeof("/Info.plist")) {
		fprintf(stderr, "runtime/cgo: bad URL: %s\n", buf);
		return;
	}
	buf[url_len-sizeof("/Info.plist")+1] = 0;
	dir = &buf[0] + sizeof("file://")-1;

	if (chdir(dir) != 0) {
		fprintf(stderr, "runtime/cgo: chdir(%s) failed\n", dir);
	}

	// The test harness in go_ios_exec passes the relative working directory
	// in the GoExecWrapperWorkingDirectory property of the app bundle.
	wd_ref = CFBundleGetValueForInfoDictionaryKey(bundle, CFSTR("GoExecWrapperWorkingDirectory"));
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

static void
init_platform()
{
#if TARGET_OS_IPHONE
	darwin_arm_init_mach_exception_handler();
	darwin_arm_init_thread_exception_port();
	init_working_dir();
#endif
}

void (*x_cgo_init_platform)(void) = init_platform;
void (*x_cgo_threadentry_platform)(void) = threadentry_platform;
