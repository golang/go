// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dashboard

// This file handles operations on the CL entity kind.

import (
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"

	"appengine"
	"appengine/datastore"
	"appengine/mail"
	"appengine/taskqueue"
	"appengine/urlfetch"
	"appengine/user"
)

func init() {
	http.HandleFunc("/assign", handleAssign)
	http.HandleFunc("/update-cl", handleUpdateCL)
}

const codereviewBase = "http://codereview.appspot.com"

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

func (cl *CL) LGTMHTML() template.HTML {
	x := make([]string, len(cl.LGTMs))
	for i, s := range cl.LGTMs {
		s = template.HTMLEscapeString(s)
		if !strings.Contains(s, "@") {
			s = "<b>" + s + "</b>"
		}
		s = `<span class="email">` + s + "</span>"
		x[i] = s
	}
	return template.HTML(strings.Join(x, ", "))
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
	if _, ok := emailToPerson[u.Email]; !ok {
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

			// We can't do this easily, as we need authentication to edit
			// an issue on behalf of a user, which is non-trivial. For now,
			// just send a mail with the body "R=<reviewer>", Cc'ing that person,
			// and rely on social convention.
			cl := new(CL)
			err := datastore.Get(c, key, cl)
			if err != nil {
				c.Errorf("%s", err)
				http.Error(w, err.Error(), 500)
				return
			}
			msg := &mail.Message{
				Sender: u.Email,
				To:     []string{preferredEmail[rev]},
				Cc:     cl.Recipients,
				// Take care to match Rietveld's subject line
				// so that Gmail will correctly thread mail.
				Subject: cl.Subject + " (issue " + n + ")",
				Body:    "R=" + rev + "\n\n(sent by gocodereview)",
			}
			// TODO(dsymonds): Use cl.LastMessageID as the In-Reply-To header
			// when the appengine/mail package supports that.
			if err := mail.Send(c, msg); err != nil {
				c.Errorf("mail.Send: %v", err)
			}
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

	url := codereviewBase + "/api/" + n + "?messages=true"
	resp, err := urlfetch.Client(c).Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("Update: got HTTP response %d", resp.StatusCode)
	}

	var apiResp struct {
		Description string `json:"description"`
		Created     string `json:"created"`
		OwnerEmail  string `json:"owner_email"`
		Modified    string `json:"modified"`
		Closed      bool   `json:"closed"`
		Subject     string `json:"subject"`
		Messages    []struct {
			Text       string   `json:"text"`
			Sender     string   `json:"sender"`
			Recipients []string `json:"recipients"`
			Approval   bool     `json:"approval"`
		} `json:"messages"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
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
	lgtm := make(map[string]bool)
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
		}

		for _, r := range msg.Recipients {
			rcpt[r] = true
		}
	}
	for l := range lgtm {
		cl.LGTMs = append(cl.LGTMs, l)
	}
	for r := range rcpt {
		cl.Recipients = append(cl.Recipients, r)
	}
	sort.Strings(cl.LGTMs)
	sort.Strings(cl.Recipients)

	key := datastore.NewKey(c, "CL", n, 0, nil)
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
