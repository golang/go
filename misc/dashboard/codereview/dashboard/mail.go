package dashboard

// This file handles receiving mail.

import (
	"net/http"
	"net/mail"
	"regexp"
	"time"

	"appengine"
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
	// Update the CL after a delay to give Rietveld a chance to catch up.
	UpdateCLLater(c, m[1], 10*time.Second)
}
