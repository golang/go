// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

#include <stdarg.h>
#ifdef __ANDROID__
#include <android/log.h>
#endif
#include "libcgo.h"

void
fatalf(const char* format, ...)
{
	va_list ap;

	fprintf(stderr, "runtime/cgo: ");
	va_start(ap, format);
	vfprintf(stderr, format, ap);
	va_end(ap);
	fprintf(stderr, "\n");

#ifdef __ANDROID__
	// When running from an Android .apk, /dev/stderr and /dev/stdout
	// redirect to /dev/null. And when running a test binary
	// via adb shell, it's easy to miss logcat. So write to both.
	va_start(ap, format);
	__android_log_vprint(ANDROID_LOG_FATAL, "runtime/cgo", format, ap);
	va_end(ap);
#endif

	abort();
}
