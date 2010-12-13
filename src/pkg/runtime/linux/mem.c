#include "runtime.h"
#include "defs.h"
#include "os.h"
#include "malloc.h"

void*
runtime·SysAlloc(uintptr n)
{
	void *p;

	mstats.sys += n;
	p = runtime·mmap(nil, n, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_PRIVATE, -1, 0);
	if(p < (void*)4096) {
		if(p == (void*)EACCES) {
			runtime·printf("mmap: access denied\n");
			runtime·printf("If you're running SELinux, enable execmem for this process.\n");
			runtime·exit(2);
		}
		runtime·printf("mmap: errno=%p\n", p);
		runtime·throw("mmap");
	}
	return p;
}

void
runtime·SysUnused(void *v, uintptr n)
{
	USED(v);
	USED(n);
	// TODO(rsc): call madvise MADV_DONTNEED
}

void
runtime·SysFree(void *v, uintptr n)
{
	mstats.sys -= n;
	runtime·munmap(v, n);
}

void
runtime·SysMemInit(void)
{
}
