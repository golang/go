#include <stdio.h>
#include "issue4339.h"

static void
impl(void)
{
	//printf("impl\n");
}

Issue4339 exported4339 = {"bar", impl};

void
handle4339(Issue4339 *x)
{
	//printf("handle\n");
	x->bar();
	//printf("done\n");
}
