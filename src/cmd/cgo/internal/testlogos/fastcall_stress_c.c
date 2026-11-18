// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

int
stress_usleep(int micros)
{
	if (micros > 0) {
#ifdef _WIN32
		Sleep((DWORD)((micros + 999) / 1000));
#else
		usleep((useconds_t)micros);
#endif
	}
	return micros;
}
