int32 runtime·thr_new(ThrParam*, int32);
void runtime·sigpanic(void);
void runtime·sigaltstack(Sigaltstack*, Sigaltstack*);
struct sigaction;
void runtime·sigaction(int32, struct sigaction*, struct sigaction*);
