// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <unistd.h>

int
stress_usleep(int micros)
{
	if (micros > 0) {
		usleep((useconds_t)micros);
	}
	return micros;
}
