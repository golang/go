// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdarg.h>
#include <android/log.h>
#include <pthread.h>
#include "libcgo.h"

void
fatalf(const char* format, ...)
{
	va_list ap;

	// Write to both stderr and logcat.
	//
	// When running from an .apk, /dev/stderr and /dev/stdout
	// redirect to /dev/null. And when running a test binary
	// via adb shell, it's easy to miss logcat.

	fprintf(stderr, "runtime/cgo: ");
	va_start(ap, format);
	vfprintf(stderr, format, ap);
	va_end(ap);
	fprintf(stderr, "\n");

	va_start(ap, format);
	__android_log_vprint(ANDROID_LOG_FATAL, "runtime/cgo", format, ap);
	va_end(ap);

	abort();
}

// Truncated to a different magic value on 32-bit; that's ok.
#define magic1 (0x23581321345589ULL)

// inittls allocates a thread-local storage slot for g.
//
// It finds the first available slot using pthread_key_create and uses
// it as the offset value for runtime.tls_g.
static void
inittls(void **tlsg, void **tlsbase)
{
	pthread_key_t k;
	int i, err;

	err = pthread_key_create(&k, nil);
	if(err != 0) {
		fatalf("pthread_key_create failed: %d", err);
	}
	pthread_setspecific(k, (void*)magic1);
	// If thread local slots are laid out as we expect, our magic word will
	// be located at some low offset from tlsbase. However, just in case something went
	// wrong, the search is limited to sensible offsets. PTHREAD_KEYS_MAX was the
	// original limit, but issue 19472 made a higher limit necessary.
	for (i=0; i<384; i++) {
		if (*(tlsbase+i) == (void*)magic1) {
			*tlsg = (void*)(i*sizeof(void *));
			pthread_setspecific(k, 0);
			return;
		}
	}
	fatalf("could not find pthread key");
}

void (*x_cgo_inittls)(void **tlsg, void **tlsbase) = inittls;
