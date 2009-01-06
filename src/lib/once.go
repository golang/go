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

type Job struct {
	done bool;
	doit chan bool;	// buffer of 1
}

type Request struct {
	f *();
	reply chan *Job
}

// TODO: Would like to use chan Request but 6g rejects it.
var service = make(chan *Request)
var jobmap = make(map[*()]*Job)

// Moderate access to the jobmap.
// Even if accesses were thread-safe (they should be but are not)
// something needs to serialize creation of new jobs.
// That's what the Server does.
func Server() {
	for {
		req := <-service;
		job, present := jobmap[req.f];
		if !present {
			job = new(Job);
			job.doit = make(chan bool, 1);
			job.doit <- true;
			jobmap[req.f] = job
		}
		req.reply <- job
	}
}

export func Do(f *()) {
	// Look for job in map (avoids channel communication).
	// If not there, ask map server to make one.
	// TODO: Uncomment use of jobmap[f] once
	// maps are thread-safe.
	var job *Job;
	var present bool;
	// job, present = jobmap[f]
	if !present {
		c := make(chan *Job);
		req := Request{f, c};
		service <- &req;
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
	go Server()
}

