// FreeBSD-specific system calls
int32 ksem_init(uint64 *, uint32);
int32 ksem_wait(uint32);
int32 ksem_destroy(uint32);
int32 ksem_post(uint32);

struct thr_param {
    void	(*start_func)(void *);	/* thread entry function. */
    void	*arg;			/* argument for entry function. */
    byte	*stack_base;		/* stack base address. */
    int64	stack_size;		/* stack size. */
    byte	*tls_base;		/* tls base address. */
    int64	tls_size;		/* tls size. */
    int64	*child_tid;		/* address to store new TID. */
    int64	*parent_tid;		/* parent accesses the new TID here. */
    int32		flags;			/* thread flags. */
    void	*spare[4];		/* TODO: cpu affinity mask etc. */
};
int32 thr_new(struct thr_param*, uint64);
