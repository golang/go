// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"appengine"
	"appengine/datastore"
	"appengine/delay"
	"appengine/mail"
	"bytes"
	"encoding/gob"
	"fmt"
	"text/template"
)

const (
	mailFrom   = "builder@golang.org" // use this for sending any mail
	failMailTo = "golang-dev@googlegroups.com"
	domain     = "build.golang.org"
)

// failIgnore is a set of builders that we don't email about because
// they're too flaky.
var failIgnore = map[string]bool{
	"netbsd-386-bsiegert":   true,
	"netbsd-amd64-bsiegert": true,
}

// notifyOnFailure checks whether the supplied Commit or the subsequent
// Commit (if present) breaks the build for this builder.
// If either of those commits break the build an email notification is sent
// from a delayed task. (We use a task because this way the mail won't be
// sent if the enclosing datastore transaction fails.)
//
// This must be run in a datastore transaction, and the provided *Commit must
// have been retrieved from the datastore within that transaction.
func notifyOnFailure(c appengine.Context, com *Commit, builder string) error {
	if failIgnore[builder] {
		return nil
	}

	// TODO(adg): implement notifications for packages
	if com.PackagePath != "" {
		return nil
	}

	p := &Package{Path: com.PackagePath}
	var broken *Commit
	cr := com.Result(builder, "")
	if cr == nil {
		return fmt.Errorf("no result for %s/%s", com.Hash, builder)
	}
	q := datastore.NewQuery("Commit").Ancestor(p.Key(c))
	if cr.OK {
		// This commit is OK. Notify if next Commit is broken.
		next := new(Commit)
		q = q.Filter("ParentHash=", com.Hash)
		if err := firstMatch(c, q, next); err != nil {
			if err == datastore.ErrNoSuchEntity {
				// OK at tip, no notification necessary.
				return nil
			}
			return err
		}
		if nr := next.Result(builder, ""); nr != nil && !nr.OK {
			c.Debugf("commit ok: %#v\nresult: %#v", com, cr)
			c.Debugf("next commit broken: %#v\nnext result:%#v", next, nr)
			broken = next
		}
	} else {
		// This commit is broken. Notify if the previous Commit is OK.
		prev := new(Commit)
		q = q.Filter("Hash=", com.ParentHash)
		if err := firstMatch(c, q, prev); err != nil {
			if err == datastore.ErrNoSuchEntity {
				// No previous result, let the backfill of
				// this result trigger the notification.
				return nil
			}
			return err
		}
		if pr := prev.Result(builder, ""); pr != nil && pr.OK {
			c.Debugf("commit broken: %#v\nresult: %#v", com, cr)
			c.Debugf("previous commit ok: %#v\nprevious result:%#v", prev, pr)
			broken = com
		}
	}
	var err error
	if broken != nil && !broken.FailNotificationSent {
		c.Infof("%s is broken commit; notifying", broken.Hash)
		sendFailMailLater.Call(c, broken, builder) // add task to queue
		broken.FailNotificationSent = true
		_, err = datastore.Put(c, broken.Key(c), broken)
	}
	return err
}

// firstMatch executes the query q and loads the first entity into v.
func firstMatch(c appengine.Context, q *datastore.Query, v interface{}) error {
	t := q.Limit(1).Run(c)
	_, err := t.Next(v)
	if err == datastore.Done {
		err = datastore.ErrNoSuchEntity
	}
	return err
}

var (
	sendFailMailLater = delay.Func("sendFailMail", sendFailMail)
	sendFailMailTmpl  = template.Must(
		template.New("notify.txt").
			Funcs(template.FuncMap(tmplFuncs)).
			ParseFiles("build/notify.txt"),
	)
)

func init() {
	gob.Register(&Commit{}) // for delay
}

// sendFailMail sends a mail notification that the build failed on the
// provided commit and builder.
func sendFailMail(c appengine.Context, com *Commit, builder string) {
	// TODO(adg): handle packages

	// get Result
	r := com.Result(builder, "")
	if r == nil {
		c.Errorf("finding result for %q: %+v", builder, com)
		return
	}

	// get Log
	k := datastore.NewKey(c, "Log", r.LogHash, 0, nil)
	l := new(Log)
	if err := datastore.Get(c, k, l); err != nil {
		c.Errorf("finding Log record %v: %v", r.LogHash, err)
		return
	}

	// prepare mail message
	var body bytes.Buffer
	err := sendFailMailTmpl.Execute(&body, map[string]interface{}{
		"Builder": builder, "Commit": com, "Result": r, "Log": l,
		"Hostname": domain,
	})
	if err != nil {
		c.Errorf("rendering mail template: %v", err)
		return
	}
	subject := fmt.Sprintf("%s broken by %s", builder, shortDesc(com.Desc))
	msg := &mail.Message{
		Sender:  mailFrom,
		To:      []string{failMailTo},
		ReplyTo: failMailTo,
		Subject: subject,
		Body:    body.String(),
	}

	// send mail
	if err := mail.Send(c, msg); err != nil {
		c.Errorf("sending mail: %v", err)
	}
}
