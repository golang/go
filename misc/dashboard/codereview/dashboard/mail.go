// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dashboard

// This file handles receiving mail.

import (
	"net/http"
	"net/mail"
	"regexp"
	"time"

	"appengine"
	"appengine/datastore"
)

func init() {
	http.HandleFunc("/_ah/mail/", handleMail)
}

var subjectRegexp = regexp.MustCompile(`.*code review (\d+):.*`)

func handleMail(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	defer r.Body.Close()
	msg, err := mail.ReadMessage(r.Body)
	if err != nil {
		c.Errorf("mail.ReadMessage: %v", err)
		return
	}

	subj := msg.Header.Get("Subject")
	m := subjectRegexp.FindStringSubmatch(subj)
	if len(m) != 2 {
		c.Debugf("Subject %q did not match /%v/", subj, subjectRegexp)
		return
	}

	c.Infof("Found issue %q", m[1])

	// Track the MessageID.
	key := datastore.NewKey(c, "CL", m[1], 0, nil)
	err = datastore.RunInTransaction(c, func(c appengine.Context) error {
		cl := new(CL)
		err := datastore.Get(c, key, cl)
		if err != nil && err != datastore.ErrNoSuchEntity {
			return err
		}
		if err == datastore.ErrNoSuchEntity {
			// Must set sentinel values for time.Time fields
			// if this is a new entity.
			cl.Created = time.Unix(0, 0)
			cl.Modified = time.Unix(0, 0)
		}
		cl.LastMessageID = msg.Header.Get("Message-ID")
		_, err = datastore.Put(c, key, cl)
		return err
	}, nil)
	if err != nil {
		c.Errorf("datastore transaction failed: %v", err)
	}

	// Update the CL after a delay to give Rietveld a chance to catch up.
	UpdateCLLater(c, m[1], 10*time.Second)
}
