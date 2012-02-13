#define SIG_DFL ((void*)0)
#define SIG_IGN ((void*)1)

int32	runtime·thr_new(ThrParam*, int32);
void	runtime·sigpanic(void);
void	runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
struct	sigaction;
void	runtime·sigaction(int32, struct sigaction*, struct sigaction*);
void	runtime·setsig(int32, void(*)(int32, Siginfo*, void*, G*), bool);
void	runtiem·setitimerval(int32, Itimerval*, Itimerval*);
void	runtime·setitimer(int32, Itimerval*, Itimerval*);
int32	runtime·sysctl(uint32*, uint32, byte*, uintptr*, byte*, uintptr);

void	runtime·raisesigpipe(void);

#define	NSIG 33
#define	SI_USER	0
