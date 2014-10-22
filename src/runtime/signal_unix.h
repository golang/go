// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_DFL ((void*)0)
#define SIG_IGN ((void*)1)

typedef void GoSighandler(int32, Siginfo*, void*, G*);
void	runtime·setsig(int32, GoSighandler*, bool);
GoSighandler* runtime·getsig(int32);

void	runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp);
void	runtime·raise(int32);

