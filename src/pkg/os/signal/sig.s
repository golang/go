// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Assembly to get into package runtime without using exported symbols.

#ifdef GOARCH_arm
#define JMP B
#endif

TEXT ·signal_enable(SB),7,$0
	JMP runtime·signal_enable(SB)

TEXT ·signal_recv(SB),7,$0
	JMP runtime·signal_recv(SB)

