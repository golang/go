// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Expose some runtime functions for testing.

typedef char bool;

bool runtime·lockedOSThread(void);

static void
FLUSH(void*)
{
}

void
·lockedOSThread(bool b)
{
	b = runtime·lockedOSThread();
	FLUSH(&b);
}
