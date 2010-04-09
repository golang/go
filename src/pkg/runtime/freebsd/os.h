int32 thr_new(ThrParam*, int32);
void sigpanic(void);
void sigaltstack(Sigaltstack*, Sigaltstack*);
struct sigaction;
void sigaction(int32, struct sigaction*, struct sigaction*);
