// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

typedef struct Flow Flow;
typedef struct Graph Graph;

struct Flow {
	Prog*	prog;   	// actual instruction
	Flow*	p1;     	// predecessors of this instruction: p1,
	Flow*	p2;     	// and then p2 linked though p2link.
	Flow*	p2link;
	Flow*	s1;     	// successors of this instruction (at most two: s1 and s2).
	Flow*	s2;
	Flow*	link;   	// next instruction in function code
	
	int32	active;	// usable by client

	int32	rpo;		// reverse post ordering
	uint16	loop;		// x5 for every loop
	uchar	refset;		// diagnostic generated
};

struct Graph
{
	Flow*	start;
	int	num;
	
	// After calling flowrpo, rpo lists the flow nodes in reverse postorder,
	// and each non-dead Flow node f has g->rpo[f->rpo] == f.
	Flow**	rpo;
};

void	fixjmp(Prog*);
Graph*	flowstart(Prog*, int);
void	flowrpo(Graph*);
void	flowend(Graph*);
void	mergetemp(Prog*);
void	nilopt(Prog*);
int	noreturn(Prog*);
int	regtyp(Addr*);
int	sameaddr(Addr*, Addr*);
int	smallindir(Addr*, Addr*);
int	stackaddr(Addr*);
Flow*	uniqp(Flow*);
Flow*	uniqs(Flow*);
