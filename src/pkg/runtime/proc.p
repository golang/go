// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
model for proc.c as of 2011/07/22.

takes 4900 seconds to explore 1189070 states
with G=3, var_gomaxprocs=1
on a Core i7 L640 2.13 GHz Lenovo X201s.

rm -f proc.p.trail pan.* pan
spin -a proc.p
gcc -DSAFETY -DREACH -DMEMLIM'='4000 -o pan pan.c
pan -w28 -n -i -m500000
test -f proc.p.trail && pan -r proc.p.trail
*/

/*
 * scheduling parameters
 */

/*
 * the number of goroutines G doubles as the maximum
 * number of OS threads; the max is reachable when all
 * the goroutines are blocked in system calls.
 */
#define G 3

/*
 * whether to allow gomaxprocs to vary during execution.
 * enabling this checks the scheduler even when code is
 * calling GOMAXPROCS, but it also slows down the verification
 * by about 10x.
 */
#define var_gomaxprocs 1  /* allow gomaxprocs to vary */

/* gomaxprocs */
#if var_gomaxprocs
byte gomaxprocs = 3;
#else
#define gomaxprocs 3
#endif

/* queue of waiting M's: sched_mhead[:mwait] */
byte mwait;
byte sched_mhead[G];

/* garbage collector state */
bit gc_lock, gcwaiting;

/* goroutines sleeping, waiting to run */
byte gsleep, gwait;

/* scheduler state */
bit sched_lock;
bit sched_stopped;
bit atomic_gwaiting, atomic_waitstop;
byte atomic_mcpu, atomic_mcpumax;

/* M struct fields - state for handing off g to m. */
bit m_waitnextg[G];
bit m_havenextg[G];
bit m_nextg[G];

/*
 * opt_atomic/opt_dstep mark atomic/deterministics
 * sequences that are marked only for reasons of
 * optimization, not for correctness of the algorithms.
 *
 * in general any code that runs while holding the
 * schedlock and does not refer to or modify the atomic_*
 * fields can be marked atomic/dstep without affecting
 * the usefulness of the model.  since we trust the lock
 * implementation, what we really want to test is the
 * interleaving of the atomic fast paths with entersyscall
 * and exitsyscall.
 */
#define opt_atomic atomic
#define opt_dstep d_step

/* locks */
inline lock(x) {
	d_step { x == 0; x = 1 }
}

inline unlock(x) {
	d_step { assert x == 1; x = 0 }
}

/* notes */
inline noteclear(x) {
	x = 0
}

inline notesleep(x) {
	x == 1
}

inline notewakeup(x) {
	opt_dstep { assert x == 0; x = 1 }
}

/*
 * scheduler
 */
inline schedlock() {
	lock(sched_lock)
}

inline schedunlock() {
	unlock(sched_lock)
}

/*
 * canaddmcpu is like the C function but takes
 * an extra argument to include in the test, to model
 * "cannget() && canaddmcpu()" as "canaddmcpu(cangget())"
 */
inline canaddmcpu(g) {
	d_step {
		g && atomic_mcpu < atomic_mcpumax;
		atomic_mcpu++;
	}
}

/*
 * gput is like the C function.
 * instead of tracking goroutines explicitly we
 * maintain only the count of the number of
 * waiting goroutines.
 */
inline gput() {
	/* omitted: lockedm, idlem concerns */
	opt_dstep {
		gwait++;
		if
		:: gwait == 1 ->
			atomic_gwaiting = 1
		:: else
		fi
	}
}

/*
 * cangget is a macro so it can be passed to
 * canaddmcpu (see above).
 */
#define cangget()  (gwait>0)

/*
 * gget is like the C function.
 */
inline gget() {
	opt_dstep {
		assert gwait > 0;
		gwait--;
		if
		:: gwait == 0 ->
			atomic_gwaiting = 0
		:: else
		fi
	}
}

/*
 * mput is like the C function.
 * here we do keep an explicit list of waiting M's,
 * so that we know which ones can be awakened.
 * we use _pid-1 because the monitor is proc 0.
 */
inline mput() {
	opt_dstep {
		sched_mhead[mwait] = _pid - 1;
		mwait++
	}
}

/*
 * mnextg is like the C function mnextg(m, g).
 * it passes an unspecified goroutine to m to start running.
 */
inline mnextg(m) {
	opt_dstep {
		m_nextg[m] = 1;
		if
		:: m_waitnextg[m] ->
			m_waitnextg[m] = 0;
			notewakeup(m_havenextg[m])
		:: else
		fi
	}
}

/*
 * mgetnextg handles the main m handoff in matchmg.
 * it is like mget() || new M followed by mnextg(m, g),
 * but combined to avoid a local variable.
 * unlike the C code, a new M simply assumes it is
 * running a g instead of using the mnextg coordination
 * to obtain one.
 */
inline mgetnextg() {
	opt_atomic {
		if
		:: mwait > 0 ->
			mwait--;
			mnextg(sched_mhead[mwait]);
			sched_mhead[mwait] = 0;
		:: else ->
			run mstart();
		fi
	}
}

/*
 * nextgandunlock is like the C function.
 * it pulls a g off the queue or else waits for one.
 */
inline nextgandunlock() {
	assert atomic_mcpu <= G;

	if
	:: m_nextg[_pid-1] ->
		m_nextg[_pid-1] = 0;
		schedunlock();
	:: canaddmcpu(!m_nextg[_pid-1] && cangget()) ->
		gget();
		schedunlock();
	:: else ->
		opt_dstep {
			mput();
			m_nextg[_pid-1] = 0;
			m_waitnextg[_pid-1] = 1;
			noteclear(m_havenextg[_pid-1]);
		}
		if
		:: atomic_waitstop && atomic_mcpu <= atomic_mcpumax ->
			atomic_waitstop = 0;
			notewakeup(sched_stopped)
		:: else
		fi;
		schedunlock();
		opt_dstep {
			notesleep(m_havenextg[_pid-1]);
			assert m_nextg[_pid-1];
			m_nextg[_pid-1] = 0;
		}
	fi
}

/*
 * stoptheworld is like the C function.
 */
inline stoptheworld() {
	schedlock();
	gcwaiting = 1;
	atomic_mcpumax = 1;
	do
	:: d_step { atomic_mcpu > 1 ->
		noteclear(sched_stopped);
		assert !atomic_waitstop;
		atomic_waitstop = 1 }
		schedunlock();
		notesleep(sched_stopped);
		schedlock();
	:: else ->
		break
	od;
	schedunlock();
}

/*
 * starttheworld is like the C function.
 */
inline starttheworld() {
	schedlock();
	gcwaiting = 0;
	atomic_mcpumax = gomaxprocs;
	matchmg();
	schedunlock();
}

/*
 * matchmg is like the C function.
 */
inline matchmg() {
	do
	:: canaddmcpu(cangget()) ->
		gget();
		mgetnextg();
	:: else -> break
	od
}

/*
 * ready is like the C function.
 * it puts a g on the run queue.
 */
inline ready() {
	schedlock();
	gput()
	matchmg()
	schedunlock()
}

/*
 * schedule simulates the C scheduler.
 * it assumes that there is always a goroutine
 * running already, and the goroutine has entered
 * the scheduler for an unspecified reason,
 * either to yield or to block.
 */
inline schedule() {
	schedlock();

	mustsched = 0;
	atomic_mcpu--;
	assert atomic_mcpu <= G;
	if
	:: skip ->
		// goroutine yields, still runnable
		gput();
	:: gsleep+1 < G ->
		// goroutine goes to sleep (but there is another that can wake it)
		gsleep++
	fi;

	// Find goroutine to run.
	nextgandunlock()
}

/*
 * schedpend is > 0 if a goroutine is about to committed to
 * entering the scheduler but has not yet done so.
 * Just as we don't test for the undesirable conditions when a
 * goroutine is in the scheduler, we don't test for them when
 * a goroutine will be in the scheduler shortly.
 * Modeling this state lets us replace mcpu cas loops with
 * simpler mcpu atomic adds.
 */
byte schedpend;

/*
 * entersyscall is like the C function.
 */
inline entersyscall() {
	bit willsched;

	/*
	 * Fast path.  Check all the conditions tested during schedlock/schedunlock
	 * below, and if we can get through the whole thing without stopping, run it
	 * in one atomic cas-based step.
	 */
	atomic {
		atomic_mcpu--;
		if
		:: atomic_gwaiting ->
			skip
		:: atomic_waitstop && atomic_mcpu <= atomic_mcpumax ->
			skip
		:: else ->
			goto Lreturn_entersyscall;
		fi;
		willsched = 1;
		schedpend++;
	}

	/*
	 * Normal path.
	 */
	schedlock()
	opt_dstep {
		if
		:: willsched ->
			schedpend--;
			willsched = 0
		:: else
		fi
	}
	if
	:: atomic_gwaiting ->
		matchmg()
	:: else
	fi;
	if
	:: atomic_waitstop && atomic_mcpu <= atomic_mcpumax ->
		atomic_waitstop = 0;
		notewakeup(sched_stopped)
	:: else
	fi;
	schedunlock();
Lreturn_entersyscall:
	skip
}

/*
 * exitsyscall is like the C function.
 */
inline exitsyscall() {
	/*
	 * Fast path.  If there's a cpu available, use it.
	 */
	atomic {
		// omitted profilehz check
		atomic_mcpu++;
		if
		:: atomic_mcpu >= atomic_mcpumax ->
			skip
		:: else ->
			goto Lreturn_exitsyscall
		fi
	}

	/*
	 * Normal path.
	 */
	schedlock();
	d_step {
		if
		:: atomic_mcpu <= atomic_mcpumax ->
			skip
		:: else ->
			mustsched = 1
		fi
	}
	schedunlock()
Lreturn_exitsyscall:
	skip
}

#if var_gomaxprocs
inline gomaxprocsfunc() {
	schedlock();
	opt_atomic {
		if
		:: gomaxprocs != 1 -> gomaxprocs = 1
		:: gomaxprocs != 2 -> gomaxprocs = 2
		:: gomaxprocs != 3 -> gomaxprocs = 3
		fi;
	}
	if
	:: gcwaiting != 0 ->
		assert atomic_mcpumax == 1
	:: else ->
		atomic_mcpumax = gomaxprocs;
		if
		:: atomic_mcpu > gomaxprocs ->
			mustsched = 1
		:: else ->
			matchmg()
		fi
	fi;
	schedunlock();
}
#endif

/*
 * mstart is the entry point for a new M.
 * our model of an M is always running some
 * unspecified goroutine.
 */
proctype mstart() {
	/*
	 * mustsched is true if the goroutine must enter the
	 * scheduler instead of continuing to execute.
	 */
	bit mustsched;

	do
	:: skip ->
		// goroutine reschedules.
		schedule()
	:: !mustsched ->
		// goroutine does something.
		if
		:: skip ->
			// goroutine executes system call
			entersyscall();
			exitsyscall()
		:: atomic { gsleep > 0; gsleep-- } ->
			// goroutine wakes another goroutine
			ready()
		:: lock(gc_lock) ->
			// goroutine runs a garbage collection
			stoptheworld();
			starttheworld();
			unlock(gc_lock)
#if var_gomaxprocs
		:: skip ->
			// goroutine picks a new gomaxprocs
			gomaxprocsfunc()
#endif
		fi
	od;

	assert 0;
}

/*
 * monitor initializes the scheduler state
 * and then watches for impossible conditions.
 */
active proctype monitor() {
	opt_dstep {
		byte i = 1;
		do
		:: i < G ->
			gput();
			i++
		:: else -> break
		od;
		atomic_mcpu = 1;
		atomic_mcpumax = 1;
	}
	run mstart();

	do
	// Should never have goroutines waiting with procs available.
	:: !sched_lock && schedpend==0 && gwait > 0 && atomic_mcpu < atomic_mcpumax ->
		assert 0
	// Should never have gc waiting for stop if things have already stopped.
	:: !sched_lock && schedpend==0 && atomic_waitstop && atomic_mcpu <= atomic_mcpumax ->
		assert 0
	od
}
