// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For one-time initialization that is not done during init.
// Wrap the initialization in a niladic function f() and call
//	once.Do(f)
// If multiple processes call once.Do(f) simultaneously
// with the same f argument, only one will call f, and the
// others will block until f finishes running.

package once

import "sync"

type job struct {
	done bool;
	sync.Mutex;	// should probably be sync.Notification or some such
}

var jobs = make(map[func()]*job)
var joblock sync.Mutex;

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
