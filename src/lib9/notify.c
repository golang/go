// +build !windows

/*
Plan 9 from User Space src/lib9/notify.c
http://code.swtch.com/plan9port/src/tip/src/lib9/notify.c

Copyright 2001-2007 Russ Cox.  All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
 * Signal handling for Plan 9 programs.
 * We stubbornly use the strings from Plan 9 instead
 * of the enumerated Unix constants.
 * There are some weird translations.  In particular,
 * a "kill" note is the same as SIGTERM in Unix.
 * There is no equivalent note to Unix's SIGKILL, since
 * it's not a deliverable signal anyway.
 *
 * We do not handle SIGABRT or SIGSEGV, mainly because
 * the thread library queues its notes for later, and we want
 * to dump core with the state at time of delivery.
 *
 * We have to add some extra entry points to provide the
 * ability to tweak which signals are deliverable and which
 * are acted upon.  Notifydisable and notifyenable play with
 * the process signal mask.  Notifyignore enables the signal
 * but will not call notifyf when it comes in.  This is occasionally
 * useful.
 */

#include <u.h>
#include <signal.h>
#define NOPLAN9DEFINES
#include <libc.h>

extern char *_p9sigstr(int, char*);
extern int _p9strsig(char*);

typedef struct Sig Sig;
struct Sig
{
	int sig;			/* signal number */
	int flags;
};

enum
{
	Restart = 1<<0,
	Ignore = 1<<1
};

static Sig sigs[] = {
	SIGHUP,		0,
	SIGINT,		0,
	SIGQUIT,		0,
	SIGILL,		0,
	SIGTRAP,		0,
/*	SIGABRT, 		0, 	*/
#ifdef SIGEMT
	SIGEMT,		0,
#endif
	SIGFPE,		0,
	SIGBUS,		0,
/*	SIGSEGV, 		0, 	*/
	SIGCHLD,		Restart|Ignore,
	SIGSYS,		0,
	SIGPIPE,		Ignore,
	SIGALRM,		0,
	SIGTERM,		0,
	SIGTSTP,		Restart|Ignore,
/*	SIGTTIN,		Restart|Ignore, */
/*	SIGTTOU,		Restart|Ignore, */
	SIGXCPU,		0,
	SIGXFSZ,		0,
	SIGVTALRM,	0,
	SIGUSR1,		0,
	SIGUSR2,		0,
#ifdef SIGWINCH
	SIGWINCH,	Restart|Ignore,
#endif
#ifdef SIGINFO
	SIGINFO,		Restart|Ignore,
#endif
};

static Sig*
findsig(int s)
{
	int i;

	for(i=0; i<nelem(sigs); i++)
		if(sigs[i].sig == s)
			return &sigs[i];
	return nil;
}

/*
 * The thread library initializes _notejmpbuf to its own
 * routine which provides a per-pthread jump buffer.
 * If we're not using the thread library, we assume we are
 * single-threaded.
 */
typedef struct Jmp Jmp;
struct Jmp
{
	p9jmp_buf b;
};

static Jmp onejmp;

static Jmp*
getonejmp(void)
{
	return &onejmp;
}

Jmp *(*_notejmpbuf)(void) = getonejmp;
static void noteinit(void);

/*
 * Actual signal handler.
 */

static void (*notifyf)(void*, char*);	/* Plan 9 handler */

static void
signotify(int sig)
{
	char tmp[64];
	Jmp *j;
	Sig *s;

	j = (*_notejmpbuf)();
	switch(p9setjmp(j->b)){
	case 0:
		if(notifyf)
			(*notifyf)(nil, _p9sigstr(sig, tmp));
		/* fall through */
	case 1:	/* noted(NDFLT) */
		if(0)print("DEFAULT %d\n", sig);
		s = findsig(sig);
		if(s && (s->flags&Ignore))
			return;
		signal(sig, SIG_DFL);
		raise(sig);
		_exit(1);
	case 2:	/* noted(NCONT) */
		if(0)print("HANDLED %d\n", sig);
		return;
	}
}

static void
signonotify(int sig)
{
	USED(sig);
}

int
noted(int v)
{
	p9longjmp((*_notejmpbuf)()->b, v==NCONT ? 2 : 1);
	abort();
	return 0;
}

int
notify(void (*f)(void*, char*))
{
	static int init;

	notifyf = f;
	if(!init){
		init = 1;
		noteinit();
	}
	return 0;
}

/*
 * Nonsense about enabling and disabling signals.
 */
typedef void Sighandler(int);
static Sighandler*
handler(int s)
{
	struct sigaction sa;

	sigaction(s, nil, &sa);
	return sa.sa_handler;
}

static int
notesetenable(int sig, int enabled)
{
	sigset_t mask, omask;

	if(sig == 0)
		return -1;

	sigemptyset(&mask);
	sigaddset(&mask, sig);
	sigprocmask(enabled ? SIG_UNBLOCK : SIG_BLOCK, &mask, &omask);
	return !sigismember(&omask, sig);
}

int
noteenable(char *msg)
{
	return notesetenable(_p9strsig(msg), 1);
}

int
notedisable(char *msg)
{
	return notesetenable(_p9strsig(msg), 0);
}

static int
notifyseton(int s, int on)
{
	Sig *sig;
	struct sigaction sa, osa;

	sig = findsig(s);
	if(sig == nil)
		return -1;
	memset(&sa, 0, sizeof sa);
	sa.sa_handler = on ? signotify : signonotify;
	if(sig->flags&Restart)
		sa.sa_flags |= SA_RESTART;

	/*
	 * We can't allow signals within signals because there's
	 * only one jump buffer.
	 */
	sigfillset(&sa.sa_mask);

	/*
	 * Install handler.
	 */
	sigaction(sig->sig, &sa, &osa);
	return osa.sa_handler == signotify;
}

int
notifyon(char *msg)
{
	return notifyseton(_p9strsig(msg), 1);
}

int
notifyoff(char *msg)
{
	return notifyseton(_p9strsig(msg), 0);
}

/*
 * Initialization follows sigs table.
 */
static void
noteinit(void)
{
	int i;
	Sig *sig;

	for(i=0; i<nelem(sigs); i++){
		sig = &sigs[i];
		/*
		 * If someone has already installed a handler,
		 * It's probably some ld preload nonsense,
		 * like pct (a SIGVTALRM-based profiler).
		 * Or maybe someone has already called notifyon/notifyoff.
		 * Leave it alone.
		 */
		if(handler(sig->sig) != SIG_DFL)
			continue;
		notifyseton(sig->sig, 1);
	}
}

