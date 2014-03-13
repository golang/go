// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"regexp"
	"text/template"

	"appengine"
	"appengine/datastore"
	"appengine/delay"
	"appengine/mail"
	"appengine/urlfetch"
)

const (
	mailFrom   = "builder@golang.org" // use this for sending any mail
	failMailTo = "golang-dev@googlegroups.com"
	domain     = "build.golang.org"
	gobotBase  = "http://research.swtch.com/gobot_codereview"
)

// ignoreFailure is a set of builders that we don't email about because
// they are not yet production-ready.
var ignoreFailure = map[string]bool{
	"dragonfly-386":           true,
	"dragonfly-amd64":         true,
	"netbsd-arm-rpi":          true,
	"solaris-amd64-smartos":   true,
	"solaris-amd64-solaris11": true,
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
	if ignoreFailure[builder] {
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
		notifyLater.Call(c, broken, builder) // add task to queue
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

var notifyLater = delay.Func("notify", notify)

// notify tries to update the CL for the given Commit with a failure message.
// If it doesn't succeed, it sends a failure email to golang-dev.
func notify(c appengine.Context, com *Commit, builder string) {
	if !updateCL(c, com, builder) {
		// Send a mail notification if the CL can't be found.
		sendFailMail(c, com, builder)
	}
}

// updateCL updates the CL for the given Commit with a failure message
// for the given builder.
func updateCL(c appengine.Context, com *Commit, builder string) bool {
	cl, err := lookupCL(c, com)
	if err != nil {
		c.Errorf("could not find CL for %v: %v", com.Hash, err)
		return false
	}
	res := com.Result(builder, "")
	if res == nil {
		c.Errorf("finding result for %q: %+v", builder, com)
		return false
	}
	url := fmt.Sprintf("%v?cl=%v&brokebuild=%v&log=%v", gobotBase, cl, builder, res.LogHash)
	r, err := urlfetch.Client(c).Post(url, "text/plain", nil)
	if err != nil {
		c.Errorf("could not update CL %v: %v", cl, err)
		return false
	}
	r.Body.Close()
	if r.StatusCode != http.StatusOK {
		c.Errorf("could not update CL %v: %v", cl, r.Status)
		return false
	}
	return true
}

var clURL = regexp.MustCompile(`https://codereview.appspot.com/([0-9]+)`)

// lookupCL consults code.google.com for the full change description for the
// provided Commit, and returns the relevant CL number.
func lookupCL(c appengine.Context, com *Commit) (string, error) {
	url := "https://code.google.com/p/go/source/detail?r=" + com.Hash
	r, err := urlfetch.Client(c).Get(url)
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return "", fmt.Errorf("retrieving %v: %v", url, r.Status)
	}
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return "", err
	}
	m := clURL.FindAllSubmatch(b, -1)
	if m == nil {
		return "", errors.New("no CL URL found on changeset page")
	}
	// Return the last visible codereview URL on the page,
	// in case the change description refers to another CL.
	return string(m[len(m)-1][1]), nil
}

var sendFailMailTmpl = template.Must(template.New("notify.txt").
	Funcs(template.FuncMap(tmplFuncs)).
	ParseFiles("build/notify.txt"))

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
