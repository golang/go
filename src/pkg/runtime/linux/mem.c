#include "runtime.h"
#include "defs.h"
#include "os.h"
#include "malloc.h"

void*
SysAlloc(uintptr n)
{
	void *p;

	mstats.sys += n;
	p = runtime_mmap(nil, n, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_PRIVATE, -1, 0);
	if(p < (void*)4096) {
		if(p == (void*)EACCES) {
			printf("mmap: access denied\n");
			printf("If you're running SELinux, enable execmem for this process.\n");
		} else {
			printf("mmap: errno=%p\n", p);
		}
		exit(2);
	}
	return p;
}

void
SysUnused(void *v, uintptr n)
{
	USED(v);
	USED(n);
	// TODO(rsc): call madvise MADV_DONTNEED
}

void
SysFree(void *v, uintptr n)
{
	USED(v);
	USED(n);
	// TODO(rsc): call munmap
}

