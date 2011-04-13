// This is stubbed out for the moment. Will revisit when the time comes.
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>

int
ctlproc(int pid, char *msg)
{
	sysfatal("ctlproc unimplemented in FreeBSD");
	return -1;
}

char*
proctextfile(int pid)
{
	sysfatal("proctextfile unimplemented in FreeBSD");
	return nil;
}

char*
procstatus(int pid)
{
	sysfatal("procstatus unimplemented in FreeBSD");
	return nil;
}

Map*
attachproc(int pid, Fhdr *fp)
{
	sysfatal("attachproc unimplemented in FreeBSD");
	return nil;
}

void
detachproc(Map *m)
{
	sysfatal("detachproc unimplemented in FreeBSD");
}

int
procthreadpids(int pid, int *p, int np)
{
	sysfatal("procthreadpids unimplemented in FreeBSD");
	return -1;
}
