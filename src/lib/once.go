// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For one-time initialization that is not done during init.
// Wrap the initialization in a niladic function f() and call
//	once.Do(&f)
// If multiple processes call once.Do(&f) simultaneously
// with the same f argument, only one will call f, and the
// others will block until f finishes running.

package once

type _Job struct {
	done bool;
	doit chan bool;	// buffer of 1
}

type _Request struct {
	f *();
	reply chan *_Job
}

var service = make(chan _Request)
var jobmap = make(map[*()]*_Job)

// Moderate access to the jobmap.
// Even if accesses were thread-safe (they should be but are not)
// something needs to serialize creation of new jobs.
// That's what the Server does.
func server() {
	for {
		req := <-service;
		job, present := jobmap[req.f];
		if !present {
			job = new(_Job);
			job.doit = make(chan bool, 1);
			job.doit <- true;
			jobmap[req.f] = job
		}
		req.reply <- job
	}
}

func Do(f *()) {
	// Look for job in map (avoids channel communication).
	// If not there, ask map server to make one.
	// TODO: Uncomment use of jobmap[f] once
	// maps are thread-safe.
	var job *_Job;
	var present bool;
	// job, present = jobmap[f]
	if !present {
		c := make(chan *_Job);
		service <- _Request{f, c};
		job = <-c
	}

	// Optimization
	if job.done {
		return
	}

	// If we're the first one, job.doit has a true waiting.
	if <-job.doit {
		f();
		job.done = true
	}

	// Leave a false waiting for the next guy.
	job.doit <- false
}

func init() {
	go server()
}

