// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dashboard

// This file handles operations on the CL entity kind.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"

	"appengine"
	"appengine/datastore"
	"appengine/taskqueue"
	"appengine/urlfetch"
	"appengine/user"
)

func init() {
	http.HandleFunc("/assign", handleAssign)
	http.HandleFunc("/update-cl", handleUpdateCL)
}

const codereviewBase = "http://codereview.appspot.com"
const gobotBase = "http://research.swtch.com/gobot_codereview"

var clRegexp = regexp.MustCompile(`\d+`)

// CL represents a code review.
type CL struct {
	Number string // e.g. "5903061"
	Closed bool
	Owner  string // email address

	Created, Modified time.Time

	Description []byte `datastore:",noindex"`
	FirstLine   string `datastore:",noindex"`
	LGTMs       []string
	NotLGTMs    []string

	// Mail information.
	Subject       string   `datastore:",noindex"`
	Recipients    []string `datastore:",noindex"`
	LastMessageID string   `datastore:",noindex"`

	// These are person IDs (e.g. "rsc"); they may be empty
	Author   string
	Reviewer string
}

// DisplayOwner returns the CL's owner, either as their email address
// or the person ID if it's a reviewer. It is for display only.
func (cl *CL) DisplayOwner() string {
	if p, ok := emailToPerson[cl.Owner]; ok {
		return p
	}
	return cl.Owner
}

func (cl *CL) FirstLineHTML() template.HTML {
	s := template.HTMLEscapeString(cl.FirstLine)
	// Embolden the package name.
	if i := strings.Index(s, ":"); i >= 0 {
		s = "<b>" + s[:i] + "</b>" + s[i:]
	}
	return template.HTML(s)
}

func formatEmails(e []string) template.HTML {
	x := make([]string, len(e))
	for i, s := range e {
		s = template.HTMLEscapeString(s)
		if !strings.Contains(s, "@") {
			s = "<b>" + s + "</b>"
		}
		s = `<span class="email">` + s + "</span>"
		x[i] = s
	}
	return template.HTML(strings.Join(x, ", "))
}

func (cl *CL) LGTMHTML() template.HTML {
	return formatEmails(cl.LGTMs)
}

func (cl *CL) NotLGTMHTML() template.HTML {
	return formatEmails(cl.NotLGTMs)
}

func (cl *CL) ModifiedAgo() string {
	// Just the first non-zero unit.
	units := [...]struct {
		suffix string
		unit   time.Duration
	}{
		{"d", 24 * time.Hour},
		{"h", time.Hour},
		{"m", time.Minute},
		{"s", time.Second},
	}
	d := time.Now().Sub(cl.Modified)
	for _, u := range units {
		if d > u.unit {
			return fmt.Sprintf("%d%s", d/u.unit, u.suffix)
		}
	}
	return "just now"
}

func handleAssign(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	if r.Method != "POST" {
		http.Error(w, "Bad method "+r.Method, 400)
		return
	}

	u := user.Current(c)
	person, ok := emailToPerson[u.Email]
	if !ok {
		http.Error(w, "Not allowed", http.StatusUnauthorized)
		return
	}

	n, rev := r.FormValue("cl"), r.FormValue("r")
	if !clRegexp.MatchString(n) {
		c.Errorf("Bad CL %q", n)
		http.Error(w, "Bad CL", 400)
		return
	}
	if _, ok := preferredEmail[rev]; !ok && rev != "" {
		c.Errorf("Unknown reviewer %q", rev)
		http.Error(w, "Unknown reviewer", 400)
		return
	}

	key := datastore.NewKey(c, "CL", n, 0, nil)

	if rev != "" {
		// Make sure the reviewer is listed in Rietveld as a reviewer.
		url := codereviewBase + "/" + n + "/fields"
		resp, err := urlfetch.Client(c).Get(url + "?field=reviewers")
		if err != nil {
			c.Errorf("Retrieving CL reviewer list failed: %v", err)
			http.Error(w, err.Error(), 500)
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			c.Errorf("Retrieving CL reviewer list failed: got HTTP response %d", resp.StatusCode)
			http.Error(w, "Failed contacting Rietveld", 500)
			return
		}

		var apiResp struct {
			Reviewers []string `json:"reviewers"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
			// probably can't be retried
			msg := fmt.Sprintf("Malformed JSON from %v: %v", url, err)
			c.Errorf("%s", msg)
			http.Error(w, msg, 500)
			return
		}
		found := false
		for _, r := range apiResp.Reviewers {
			if emailToPerson[r] == rev {
				found = true
				break
			}
		}
		if !found {
			c.Infof("Adding %v as a reviewer of CL %v", rev, n)

			url := fmt.Sprintf("%s?cl=%s&r=%s&obo=%s", gobotBase, n, rev, person)
			resp, err := urlfetch.Client(c).Get(url)
			if err != nil {
				c.Errorf("Gobot GET failed: %v", err)
				http.Error(w, err.Error(), 500)
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != 200 {
				c.Errorf("Gobot GET failed: got HTTP response %d", resp.StatusCode)
				http.Error(w, "Failed contacting Gobot", 500)
				return
			}

			c.Infof("Gobot said %q", resp.Status)
		}
	}

	// Update our own record.
	err := datastore.RunInTransaction(c, func(c appengine.Context) error {
		cl := new(CL)
		err := datastore.Get(c, key, cl)
		if err != nil {
			return err
		}
		cl.Reviewer = rev
		_, err = datastore.Put(c, key, cl)
		return err
	}, nil)
	if err != nil {
		msg := fmt.Sprintf("Assignment failed: %v", err)
		c.Errorf("%s", msg)
		http.Error(w, msg, 500)
		return
	}
	c.Infof("Assigned CL %v to %v", n, rev)
}

func UpdateCLLater(c appengine.Context, n string, delay time.Duration) {
	t := taskqueue.NewPOSTTask("/update-cl", url.Values{
		"cl": []string{n},
	})
	t.Delay = delay
	if _, err := taskqueue.Add(c, t, "update-cl"); err != nil {
		c.Errorf("Failed adding task: %v", err)
	}
}

func handleUpdateCL(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	n := r.FormValue("cl")
	if !clRegexp.MatchString(n) {
		c.Errorf("Bad CL %q", n)
		http.Error(w, "Bad CL", 400)
		return
	}

	if err := updateCL(c, n); err != nil {
		c.Errorf("Failed updating CL %v: %v", n, err)
		http.Error(w, "Failed update", 500)
		return
	}

	io.WriteString(w, "OK")
}

// updateCL updates a single CL. If a retryable failure occurs, an error is returned.
func updateCL(c appengine.Context, n string) error {
	c.Debugf("Updating CL %v", n)
	key := datastore.NewKey(c, "CL", n, 0, nil)

	url := codereviewBase + "/api/" + n + "?messages=true"
	resp, err := urlfetch.Client(c).Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	raw, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("Failed reading HTTP body: %v", err)
	}

	// Special case for abandoned CLs.
	if resp.StatusCode == 404 && bytes.Contains(raw, []byte("No issue exists with that id")) {
		// Don't bother checking for errors. The CL might never have been saved, for instance.
		datastore.Delete(c, key)
		c.Infof("Deleted abandoned CL %v", n)
		return nil
	}

	if resp.StatusCode != 200 {
		return fmt.Errorf("Update: got HTTP response %d", resp.StatusCode)
	}

	var apiResp struct {
		Description string   `json:"description"`
		Reviewers   []string `json:"reviewers"`
		Created     string   `json:"created"`
		OwnerEmail  string   `json:"owner_email"`
		Modified    string   `json:"modified"`
		Closed      bool     `json:"closed"`
		Subject     string   `json:"subject"`
		Messages    []struct {
			Text       string   `json:"text"`
			Sender     string   `json:"sender"`
			Recipients []string `json:"recipients"`
			Approval   bool     `json:"approval"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &apiResp); err != nil {
		// probably can't be retried
		c.Errorf("Malformed JSON from %v: %v", url, err)
		return nil
	}
	//c.Infof("RAW: %+v", apiResp)

	cl := &CL{
		Number:      n,
		Closed:      apiResp.Closed,
		Owner:       apiResp.OwnerEmail,
		Description: []byte(apiResp.Description),
		FirstLine:   apiResp.Description,
		Subject:     apiResp.Subject,
		Author:      emailToPerson[apiResp.OwnerEmail],
	}
	cl.Created, err = time.Parse("2006-01-02 15:04:05.000000", apiResp.Created)
	if err != nil {
		c.Errorf("Bad creation time %q: %v", apiResp.Created, err)
	}
	cl.Modified, err = time.Parse("2006-01-02 15:04:05.000000", apiResp.Modified)
	if err != nil {
		c.Errorf("Bad modification time %q: %v", apiResp.Modified, err)
	}
	if i := strings.Index(cl.FirstLine, "\n"); i >= 0 {
		cl.FirstLine = cl.FirstLine[:i]
	}
	// Treat zero reviewers as a signal that the CL is completed.
	// This could be after the CL has been submitted, but before the CL author has synced,
	// but it could also be a CL manually edited to remove reviewers.
	if len(apiResp.Reviewers) == 0 {
		cl.Closed = true
	}

	lgtm := make(map[string]bool)
	notLGTM := make(map[string]bool)
	rcpt := make(map[string]bool)
	for _, msg := range apiResp.Messages {
		s, rev := msg.Sender, false
		if p, ok := emailToPerson[s]; ok {
			s, rev = p, true
		}

		// CLs submitted by someone other than the CL owner do not immediately
		// transition to "closed". Let's simulate the intention by treating
		// messages starting with "*** Submitted as " from a reviewer as a
		// signal that the CL is now closed.
		if rev && strings.HasPrefix(msg.Text, "*** Submitted as ") {
			cl.Closed = true
		}

		if msg.Approval {
			lgtm[s] = true
			delete(notLGTM, s) // "LGTM" overrules previous "NOT LGTM"
		}
		if strings.Contains(msg.Text, "NOT LGTM") {
			notLGTM[s] = true
			delete(lgtm, s) // "NOT LGTM" overrules previous "LGTM"
		}

		for _, r := range msg.Recipients {
			rcpt[r] = true
		}
	}
	for l := range lgtm {
		cl.LGTMs = append(cl.LGTMs, l)
	}
	for l := range notLGTM {
		cl.NotLGTMs = append(cl.NotLGTMs, l)
	}
	for r := range rcpt {
		cl.Recipients = append(cl.Recipients, r)
	}
	sort.Strings(cl.LGTMs)
	sort.Strings(cl.NotLGTMs)
	sort.Strings(cl.Recipients)

	err = datastore.RunInTransaction(c, func(c appengine.Context) error {
		ocl := new(CL)
		err := datastore.Get(c, key, ocl)
		if err != nil && err != datastore.ErrNoSuchEntity {
			return err
		} else if err == nil {
			// LastMessageID and Reviewer need preserving.
			cl.LastMessageID = ocl.LastMessageID
			cl.Reviewer = ocl.Reviewer
		}
		_, err = datastore.Put(c, key, cl)
		return err
	}, nil)
	if err != nil {
		return err
	}
	c.Infof("Updated CL %v", n)
	return nil
}
