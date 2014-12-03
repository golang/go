// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"encoding/json"
	"fmt"
	"net/http"

	"appengine"
	"appengine/datastore"
)

func init() {
	handleFunc("/updatebenchmark", updateBenchmark)
}

func updateBenchmark(w http.ResponseWriter, r *http.Request) {
	if !appengine.IsDevAppServer() {
		fmt.Fprint(w, "Update must not run on real server.")
		return
	}

	if r.Method != "POST" {
		fmt.Fprintf(w, "bad request method")
		return
	}

	c := contextForRequest(r)
	if !validKey(c, r.FormValue("key"), r.FormValue("builder")) {
		fmt.Fprintf(w, "bad builder/key")
		return
	}

	defer r.Body.Close()
	var hashes []string
	if err := json.NewDecoder(r.Body).Decode(&hashes); err != nil {
		fmt.Fprintf(w, "failed to decode request: %v", err)
		return
	}

	ncommit := 0
	nrun := 0
	tx := func(c appengine.Context) error {
		var cr *CommitRun
		for _, hash := range hashes {
			// Update Commit.
			com := &Commit{Hash: hash}
			err := datastore.Get(c, com.Key(c), com)
			if err != nil && err != datastore.ErrNoSuchEntity {
				return fmt.Errorf("fetching Commit: %v", err)
			}
			if err == datastore.ErrNoSuchEntity {
				continue
			}
			com.NeedsBenchmarking = true
			com.PerfResults = nil
			if err := putCommit(c, com); err != nil {
				return err
			}
			ncommit++

			// create PerfResult
			res := &PerfResult{CommitHash: com.Hash, CommitNum: com.Num}
			err = datastore.Get(c, res.Key(c), res)
			if err != nil && err != datastore.ErrNoSuchEntity {
				return fmt.Errorf("fetching PerfResult: %v", err)
			}
			if err == datastore.ErrNoSuchEntity {
				if _, err := datastore.Put(c, res.Key(c), res); err != nil {
					return fmt.Errorf("putting PerfResult: %v", err)
				}
			}

			// Update CommitRun.
			if cr != nil && cr.StartCommitNum != com.Num/PerfRunLength*PerfRunLength {
				if _, err := datastore.Put(c, cr.Key(c), cr); err != nil {
					return fmt.Errorf("putting CommitRun: %v", err)
				}
				nrun++
				cr = nil
			}
			if cr == nil {
				var err error
				cr, err = GetCommitRun(c, com.Num)
				if err != nil {
					return fmt.Errorf("getting CommitRun: %v", err)
				}
			}
			if com.Num < cr.StartCommitNum || com.Num >= cr.StartCommitNum+PerfRunLength {
				return fmt.Errorf("commit num %v out of range [%v, %v)", com.Num, cr.StartCommitNum, cr.StartCommitNum+PerfRunLength)
			}
			idx := com.Num - cr.StartCommitNum
			cr.Hash[idx] = com.Hash
			cr.User[idx] = shortDesc(com.User)
			cr.Desc[idx] = shortDesc(com.Desc)
			cr.Time[idx] = com.Time
			cr.NeedsBenchmarking[idx] = com.NeedsBenchmarking
		}
		if cr != nil {
			if _, err := datastore.Put(c, cr.Key(c), cr); err != nil {
				return fmt.Errorf("putting CommitRun: %v", err)
			}
			nrun++
		}
		return nil
	}
	if err := datastore.RunInTransaction(c, tx, nil); err != nil {
		fmt.Fprintf(w, "failed to execute tx: %v", err)
		return
	}
	fmt.Fprintf(w, "OK (updated %v commits and %v commit runs)", ncommit, nrun)
}
