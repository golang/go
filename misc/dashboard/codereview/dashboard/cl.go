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
	units := map[string]time.Duration{
		"d": 24 * time.Hour,
		"h": time.Hour,
		"m": time.Minute,
		"s": time.Second,
	}
	d := time.Now().Sub(cl.Modified)
	for suffix, u := range units {
		if d > u {
			return fmt.Sprintf("%d%s", d/u, suffix)
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

	if _, ok := emailToPerson[user.Current(c).Email]; !ok {
		http.Error(w, "Not allowed", http.StatusUnauthorized)
		return
	}

	n, rev := r.FormValue("cl"), r.FormValue("r")
	if !clRegexp.MatchString(n) {
		c.Errorf("Bad CL %q", n)
		http.Error(w, "Bad CL", 400)
		return
	}

	key := datastore.NewKey(c, "CL", n, 0, nil)
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
		Messages    []struct {
			Text     string `json:"text"`
			Sender   string `json:"sender"`
			Approval bool   `json:"approval"`
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
			cl.LGTMs = append(cl.LGTMs, s)
		}
	}
	sort.Strings(cl.LGTMs)

	key := datastore.NewKey(c, "CL", n, 0, nil)
	err = datastore.RunInTransaction(c, func(c appengine.Context) error {
		ocl := new(CL)
		err := datastore.Get(c, key, ocl)
		if err != nil && err != datastore.ErrNoSuchEntity {
			return err
		} else if err == nil {
			// Reviewer is the only field that needs preserving.
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
