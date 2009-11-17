// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

void	bsdthread_create(void*, M*, G*, void(*)(void));
void	bsdthread_register(void);
int32	mach_msg_trap(MachHeader*, int32, uint32, uint32, uint32, uint32, uint32);
uint32	mach_reply_port(void);
void	mach_semacquire(uint32);
uint32	mach_semcreate(void);
void	mach_semdestroy(uint32);
void	mach_semrelease(uint32);
void	mach_semreset(uint32);
uint32	mach_task_self(void);
uint32	mach_task_self(void);
uint32	mach_thread_self(void);
uint32	mach_thread_self(void);

struct Sigaction;
void	sigaction(uintptr, struct Sigaction*, struct Sigaction*);

struct StackT;
void	sigaltstack(struct StackT*, struct StackT*);
void	sigtramp(void);
