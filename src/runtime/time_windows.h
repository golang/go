// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Constants for fetching time values on Windows for use in asm code.

// See https://wrkhpi.wordpress.com/2007/08/09/getting-os-information-the-kuser_shared_data-structure/
// Archived copy at:
// http://web.archive.org/web/20210411000829/https://wrkhpi.wordpress.com/2007/08/09/getting-os-information-the-kuser_shared_data-structure/

// Must read hi1, then lo, then hi2. The snapshot is valid if hi1 == hi2.
// Or, on 64-bit, just read lo:hi1 all at once atomically.
#define _INTERRUPT_TIME 0x7ffe0008
#define _SYSTEM_TIME 0x7ffe0014
#define time_lo 0
#define time_hi1 4
#define time_hi2 8
