// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Definitions related to data race detection.

#ifdef RACE
enum { raceenabled = 1 };
#else
enum { raceenabled = 0 };
#endif

// Initialize race detection subsystem.
uintptr	runtime·raceinit(void);
// Finalize race detection subsystem, does not return.
void	runtime·racefini(void);

void	runtime·racemapshadow(void *addr, uintptr size);
void	runtime·racemalloc(void *p, uintptr sz);
uintptr	runtime·racegostart(void *pc);
void	runtime·racegoend(void);
void	runtime·racewritepc(void *addr, void *callpc, void *pc);
void	runtime·racereadpc(void *addr, void *callpc, void *pc);
void	runtime·racewriterangepc(void *addr, uintptr sz, void *callpc, void *pc);
void	runtime·racereadrangepc(void *addr, uintptr sz, void *callpc, void *pc);
void	runtime·racereadobjectpc(void *addr, Type *t, void *callpc, void *pc);
void	runtime·racewriteobjectpc(void *addr, Type *t, void *callpc, void *pc);
void	runtime·racefingo(void);
void	runtime·raceacquire(void *addr);
void	runtime·raceacquireg(G *gp, void *addr);
void	runtime·racerelease(void *addr);
void	runtime·racereleaseg(G *gp, void *addr);
void	runtime·racereleasemerge(void *addr);
void	runtime·racereleasemergeg(G *gp, void *addr);
