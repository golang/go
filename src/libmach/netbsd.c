// This is stubbed out for the moment. Will revisit when the time comes.
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>

int
ctlproc(int pid, char *msg)
{
	sysfatal("ctlproc unimplemented in NetBSD");
	return -1;
}

char*
proctextfile(int pid)
{
	sysfatal("proctextfile unimplemented in NetBSD");
	return nil;
}

char*
procstatus(int pid)
{
	sysfatal("procstatus unimplemented in NetBSD");
	return nil;
}

Map*
attachproc(int pid, Fhdr *fp)
{
	sysfatal("attachproc unimplemented in NetBSD");
	return nil;
}

void
detachproc(Map *m)
{
	sysfatal("detachproc unimplemented in NetBSD");
}

int
procthreadpids(int pid, int *p, int np)
{
	sysfatal("procthreadpids unimplemented in NetBSD");
	return -1;
}
