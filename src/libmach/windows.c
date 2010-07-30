// This is stubbed out for the moment. Will revisit when the time comes.
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>

int
ctlproc(int pid, char *msg)
{
	sysfatal("ctlproc unimplemented in Windows");
}

char*
proctextfile(int pid)
{
	sysfatal("proctextfile unimplemented in Windows");
}

char*
procstatus(int pid)
{
	sysfatal("procstatus unimplemented in Windows");
}

Map*
attachproc(int pid, Fhdr *fp)
{
	sysfatal("attachproc unimplemented in Windows");
}

void
detachproc(Map *m)
{
	sysfatal("detachproc unimplemented in Windows");
}

int
procthreadpids(int pid, int *p, int np)
{
	sysfatal("procthreadpids unimplemented in Windows");
}

int 
pread(int fd, void *buf, int count, int offset)
{
	sysfatal("pread unimplemented in Windows");
}

int 
pwrite(int fd, void *buf, int count, int offset)
{
	sysfatal("pwrite unimplemented in Windows");
}

int 
nanosleep(const struct timespec *rqtp, struct timespec *rmtp)
{
	sysfatal("nanosleep unimplemented in Windows");
}
