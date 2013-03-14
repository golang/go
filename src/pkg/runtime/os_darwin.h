// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SS_DISABLE 4

int32	runtime·bsdthread_create(void*, M*, G*, void(*)(void));
int32	runtime·bsdthread_register(void);
int32	runtime·mach_msg_trap(MachHeader*, int32, uint32, uint32, uint32, uint32, uint32);
uint32	runtime·mach_reply_port(void);
int32	runtime·mach_semacquire(uint32, int64);
uint32	runtime·mach_semcreate(void);
void	runtime·mach_semdestroy(uint32);
void	runtime·mach_semrelease(uint32);
void	runtime·mach_semreset(uint32);
uint32	runtime·mach_task_self(void);
uint32	runtime·mach_task_self(void);
uint32	runtime·mach_thread_self(void);
uint32	runtime·mach_thread_self(void);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);

typedef uint32 Sigset;
void	runtime·sigprocmask(int32, Sigset*, Sigset*);

struct Sigaction;
void	runtime·sigaction(uintptr, struct Sigaction*, struct Sigaction*);

struct StackT;
void	runtime·sigaltstack(struct StackT*, struct StackT*);
void	runtime·sigtramp(void);
void	runtime·sigpanic(void);
void	runtime·setitimer(int32, Itimerval*, Itimerval*);


#define	NSIG 32
#define	SI_USER	0  /* empirically true, but not what headers say */
#define	SIG_BLOCK 1
#define	SIG_UNBLOCK 2
#define	SIG_SETMASK 3

