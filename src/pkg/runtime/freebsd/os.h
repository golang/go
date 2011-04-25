#define SIG_DFL ((void*)0)
#define SIG_IGN ((void*)1)

int32 runtime·thr_new(ThrParam*, int32);
void runtime·sigpanic(void);
void runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
struct sigaction;
void runtime·sigaction(int32, struct sigaction*, struct sigaction*);
void	runtiem·setitimerval(int32, Itimerval*, Itimerval*);
void	runtime·setitimer(int32, Itimerval*, Itimerval*);

void	runtime·raisesigpipe(void);
