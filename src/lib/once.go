// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package provides a single function, Do, to run a function
// exactly once, usually used as part of initialization.
package once

import "sync"

type job struct {
	done bool;
	sync.Mutex;	// should probably be sync.Notification or some such
}

var jobs = make(map[func()]*job)
var joblock sync.Mutex;

// Do is the the only exported piece of the package.
// For one-time initialization that is not done during init,
// wrap the initialization in a niladic function f() and call
//	Do(f)
// If multiple processes call Do(f) simultaneously
// with the same f argument, only one will call f, and the
// others will block until f finishes running.
func Do(f func()) {
	joblock.Lock();
	j, present := jobs[f];
	if !present {
		// run it
		j = new(job);
		j.Lock();
		jobs[f] = j;
		joblock.Unlock();
		f();
		j.done = true;
		j.Unlock();
	} else {
		// wait for it
		joblock.Unlock();
		if j.done != true {
			j.Lock();
			j.Unlock();
		}
	}
}
