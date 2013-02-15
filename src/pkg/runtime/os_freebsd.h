#define SIG_DFL ((void*)0)
#define SIG_IGN ((void*)1)
#define SIGHUP 1

int32	runtime·thr_new(ThrParam*, int32);
void	runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp);
void	runtime·sigpanic(void);
void	runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
struct	sigaction;
void	runtime·sigaction(int32, struct sigaction*, struct sigaction*);
void	runtime·sigprocmask(Sigset *, Sigset *);
void	runtime·setsig(int32, void(*)(int32, Siginfo*, void*, G*), bool);
void	runtime·setitimer(int32, Itimerval*, Itimerval*);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);

void	runtime·raisesigpipe(void);

#define	NSIG 33
#define	SI_USER	0x10001

#define RLIMIT_AS 10
typedef struct Rlimit Rlimit;
struct Rlimit {
	int64	rlim_cur;
	int64	rlim_max;
};
int32	runtime·getrlimit(int32, Rlimit*);
