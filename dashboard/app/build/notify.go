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
	"net/url"
	"regexp"
	"runtime"
	"sort"
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
	"dragonfly-386":         true,
	"dragonfly-amd64":       true,
	"freebsd-arm":           true,
	"netbsd-amd64-bsiegert": true,
	"netbsd-arm-rpi":        true,
	"plan9-amd64-aram":      true,
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
	if broken == nil {
		return nil
	}
	r := broken.Result(builder, "")
	if r == nil {
		return fmt.Errorf("finding result for %q: %+v", builder, com)
	}
	return commonNotify(c, broken, builder, r.LogHash)
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
func notify(c appengine.Context, com *Commit, builder, logHash string) {
	v := url.Values{"brokebuild": {builder}, "log": {logHash}}
	if !updateCL(c, com, v) {
		// Send a mail notification if the CL can't be found.
		sendFailMail(c, com, builder, logHash)
	}
}

// updateCL tells gobot to update the CL for the given Commit with
// the provided query values.
func updateCL(c appengine.Context, com *Commit, v url.Values) bool {
	cl, err := lookupCL(c, com)
	if err != nil {
		c.Errorf("could not find CL for %v: %v", com.Hash, err)
		return false
	}
	u := fmt.Sprintf("%v?cl=%v&%s", gobotBase, cl, v.Encode())
	r, err := urlfetch.Client(c).Post(u, "text/plain", nil)
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

var (
	sendPerfMailLater = delay.Func("sendPerfMail", sendPerfMailFunc)
	sendPerfMailTmpl  = template.Must(
		template.New("perf_notify.txt").
			Funcs(template.FuncMap(tmplFuncs)).
			ParseFiles("build/perf_notify.txt"),
	)
)

// MUST be called from inside a transaction.
func sendPerfFailMail(c appengine.Context, builder string, res *PerfResult) error {
	com := &Commit{Hash: res.CommitHash}
	if err := datastore.Get(c, com.Key(c), com); err != nil {
		return err
	}
	logHash := ""
	parsed := res.ParseData()
	for _, data := range parsed[builder] {
		if !data.OK {
			logHash = data.Artifacts["log"]
			break
		}
	}
	if logHash == "" {
		return fmt.Errorf("can not find failed result for commit %v on builder %v", com.Hash, builder)
	}
	return commonNotify(c, com, builder, logHash)
}

// commonNotify MUST!!! be called from within a transaction inside which
// the provided Commit entity was retrieved from the datastore.
func commonNotify(c appengine.Context, com *Commit, builder, logHash string) error {
	if com.Num == 0 || com.Desc == "" {
		stk := make([]byte, 10000)
		n := runtime.Stack(stk, false)
		stk = stk[:n]
		c.Errorf("refusing to notify with com=%+v\n%s", *com, string(stk))
		return fmt.Errorf("misuse of commonNotify")
	}
	if com.FailNotificationSent {
		return nil
	}
	c.Infof("%s is broken commit; notifying", com.Hash)
	notifyLater.Call(c, com, builder, logHash) // add task to queue
	com.FailNotificationSent = true
	return putCommit(c, com)
}

// sendFailMail sends a mail notification that the build failed on the
// provided commit and builder.
func sendFailMail(c appengine.Context, com *Commit, builder, logHash string) {
	// get Log
	k := datastore.NewKey(c, "Log", logHash, 0, nil)
	l := new(Log)
	if err := datastore.Get(c, k, l); err != nil {
		c.Errorf("finding Log record %v: %v", logHash, err)
		return
	}
	logText, err := l.Text()
	if err != nil {
		c.Errorf("unpacking Log record %v: %v", logHash, err)
		return
	}

	// prepare mail message
	var body bytes.Buffer
	err = sendFailMailTmpl.Execute(&body, map[string]interface{}{
		"Builder": builder, "Commit": com, "LogHash": logHash, "LogText": logText,
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

type PerfChangeBenchmark struct {
	Name    string
	Metrics []*PerfChangeMetric
}

type PerfChangeMetric struct {
	Name  string
	Old   uint64
	New   uint64
	Delta float64
}

type PerfChangeBenchmarkSlice []*PerfChangeBenchmark

func (l PerfChangeBenchmarkSlice) Len() int      { return len(l) }
func (l PerfChangeBenchmarkSlice) Swap(i, j int) { l[i], l[j] = l[j], l[i] }
func (l PerfChangeBenchmarkSlice) Less(i, j int) bool {
	b1, p1 := splitBench(l[i].Name)
	b2, p2 := splitBench(l[j].Name)
	if b1 != b2 {
		return b1 < b2
	}
	return p1 < p2
}

type PerfChangeMetricSlice []*PerfChangeMetric

func (l PerfChangeMetricSlice) Len() int           { return len(l) }
func (l PerfChangeMetricSlice) Swap(i, j int)      { l[i], l[j] = l[j], l[i] }
func (l PerfChangeMetricSlice) Less(i, j int) bool { return l[i].Name < l[j].Name }

func sendPerfMailFunc(c appengine.Context, com *Commit, prevCommitHash, builder string, changes []*PerfChange) {
	// Sort the changes into the right order.
	var benchmarks []*PerfChangeBenchmark
	for _, ch := range changes {
		// Find the benchmark.
		var b *PerfChangeBenchmark
		for _, b1 := range benchmarks {
			if b1.Name == ch.Bench {
				b = b1
				break
			}
		}
		if b == nil {
			b = &PerfChangeBenchmark{Name: ch.Bench}
			benchmarks = append(benchmarks, b)
		}
		b.Metrics = append(b.Metrics, &PerfChangeMetric{Name: ch.Metric, Old: ch.Old, New: ch.New, Delta: ch.Diff})
	}
	for _, b := range benchmarks {
		sort.Sort(PerfChangeMetricSlice(b.Metrics))
	}
	sort.Sort(PerfChangeBenchmarkSlice(benchmarks))

	u := fmt.Sprintf("http://%v/perfdetail?commit=%v&commit0=%v&kind=builder&builder=%v", domain, com.Hash, prevCommitHash, builder)

	// Prepare mail message (without Commit, for updateCL).
	var body bytes.Buffer
	err := sendPerfMailTmpl.Execute(&body, map[string]interface{}{
		"Builder": builder, "Hostname": domain, "Url": u, "Benchmarks": benchmarks,
	})
	if err != nil {
		c.Errorf("rendering perf mail template: %v", err)
		return
	}

	// First, try to update the CL.
	v := url.Values{"textmsg": {body.String()}}
	if updateCL(c, com, v) {
		return
	}

	// Otherwise, send mail (with Commit, for independent mail message).
	body.Reset()
	err = sendPerfMailTmpl.Execute(&body, map[string]interface{}{
		"Builder": builder, "Commit": com, "Hostname": domain, "Url": u, "Benchmarks": benchmarks,
	})
	if err != nil {
		c.Errorf("rendering perf mail template: %v", err)
		return
	}
	subject := fmt.Sprintf("Perf changes on %s by %s", builder, shortDesc(com.Desc))
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
