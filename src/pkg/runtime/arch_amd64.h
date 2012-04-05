enum {
	thechar = '6',
	CacheLineSize = 64
};

// prefetches *addr into processor's cache
#define PREFETCH(addr) runtime·prefetch(addr)
void	runtime·prefetch(void*);
