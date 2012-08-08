// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dashboard

// This file handles the front page.

import (
	"bytes"
	"html/template"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"appengine"
	"appengine/datastore"
	"appengine/user"
)

func init() {
	http.HandleFunc("/", handleFront)
	http.HandleFunc("/favicon.ico", http.NotFound)
}

// maximum number of active CLs to show in person-specific tables.
const maxCLs = 100

func handleFront(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	data := &frontPageData{
		Reviewers: personList,
		User:      user.Current(c).Email,
		IsAdmin:   user.IsAdmin(c),
	}
	var currentPerson string
	u := data.User
	you := "you"
	if e := r.FormValue("user"); e != "" {
		u = e
		you = e
	}
	currentPerson, data.UserIsReviewer = emailToPerson[u]
	if !data.UserIsReviewer {
		currentPerson = u
	}

	var wg sync.WaitGroup
	errc := make(chan error, 10)
	activeCLs := datastore.NewQuery("CL").
		Filter("Closed =", false).
		Order("-Modified")

	tableFetch := func(index int, f func(tbl *clTable) error) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			if err := f(&data.Tables[index]); err != nil {
				errc <- err
			}
			data.Timing[index] = time.Now().Sub(start)
		}()
	}

	data.Tables[0].Title = "CLs assigned to " + you + " for review"
	if data.UserIsReviewer {
		tableFetch(0, func(tbl *clTable) error {
			q := activeCLs.Filter("Reviewer =", currentPerson).Limit(maxCLs)
			tbl.Assignable = true
			_, err := q.GetAll(c, &tbl.CLs)
			return err
		})
	}

	tableFetch(1, func(tbl *clTable) error {
		q := activeCLs
		if data.UserIsReviewer {
			q = q.Filter("Author =", currentPerson)
		} else {
			q = q.Filter("Owner =", currentPerson)
		}
		q = q.Limit(maxCLs)
		tbl.Title = "CLs sent by " + you
		tbl.Assignable = true
		_, err := q.GetAll(c, &tbl.CLs)
		return err
	})

	tableFetch(2, func(tbl *clTable) error {
		q := activeCLs.Limit(50)
		tbl.Title = "Other active CLs"
		tbl.Assignable = true
		if _, err := q.GetAll(c, &tbl.CLs); err != nil {
			return err
		}
		// filter
		for i := len(tbl.CLs) - 1; i >= 0; i-- {
			cl := tbl.CLs[i]
			if cl.Owner == currentPerson || cl.Author == currentPerson || cl.Reviewer == currentPerson {
				// Preserve order.
				copy(tbl.CLs[i:], tbl.CLs[i+1:])
				tbl.CLs = tbl.CLs[:len(tbl.CLs)-1]
			}
		}
		return nil
	})

	tableFetch(3, func(tbl *clTable) error {
		q := datastore.NewQuery("CL").
			Filter("Closed =", true).
			Order("-Modified").
			Limit(10)
		tbl.Title = "Recently closed CLs"
		tbl.Assignable = false
		_, err := q.GetAll(c, &tbl.CLs)
		return err
	})

	// Not really a table fetch.
	tableFetch(0, func(_ *clTable) error {
		var err error
		data.LogoutURL, err = user.LogoutURL(c, "/")
		return err
	})

	wg.Wait()

	select {
	case err := <-errc:
		c.Errorf("%v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	default:
	}

	var b bytes.Buffer
	if err := frontPage.ExecuteTemplate(&b, "front", &data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	io.Copy(w, &b)
}

type frontPageData struct {
	Tables [4]clTable
	Timing [4]time.Duration

	Reviewers      []string
	UserIsReviewer bool

	User, LogoutURL string // actual logged in user
	IsAdmin         bool
}

type clTable struct {
	Title      string
	Assignable bool
	CLs        []*CL
}

var frontPage = template.Must(template.New("front").Funcs(template.FuncMap{
	"selected": func(a, b string) string {
		if a == b {
			return "selected"
		}
		return ""
	},
	"shortemail": func(s string) string {
		if i := strings.Index(s, "@"); i >= 0 {
			s = s[:i]
		}
		return s
	},
}).Parse(`
<!doctype html>
<html>
  <head>
    <title>Go code reviews</title>
    <link rel="icon" type="image/png" href="/static/icon.png" />
    <style type="text/css">
      body {
        font-family: Helvetica, sans-serif;
      }
      img#gopherstamp {
        float: right;
	height: auto;
	width: 250px;
      }
      h1, h2, h3 {
        color: #777;
	margin-bottom: 0;
      }
      table {
        border-spacing: 0;
      }
      td {
        vertical-align: top;
        padding: 2px 5px;
      }
      tr.unreplied td.email {
        border-left: 2px solid blue;
      }
      tr.pending td {
        background: #fc8;
      }
      tr.failed td {
        background: #f88;
      }
      tr.saved td {
        background: #8f8;
      }
      .cls {
        margin-top: 0;
      }
      a {
        color: blue;
	text-decoration: none;  /* no link underline */
      }
      address {
        font-size: 10px;
	text-align: right;
      }
      .email {
        font-family: monospace;
      }
    </style>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.2/jquery.min.js"></script>
  <head>
  <body>

<img id="gopherstamp" src="/static/gopherstamp.jpg" />
<h1>Go code reviews</h1>

<table class="cls">
{{range $i, $tbl := .Tables}}
<tr><td colspan="5"><h3>{{$tbl.Title}}</h3></td></tr>
{{if .CLs}}
{{range $cl := .CLs}}
  <tr id="cl-{{$cl.Number}}" class="{{if not $i}}{{if not .Reviewed}}unreplied{{end}}{{end}}">
    <td class="email">{{$cl.DisplayOwner}}</td>
    <td>
    {{if $tbl.Assignable}}
    <select id="cl-rev-{{$cl.Number}}" {{if not $.UserIsReviewer}}disabled{{end}}>
      <option></option>
      {{range $.Reviewers}}
      <option {{selected . $cl.Reviewer}}>{{.}}</option>
      {{end}}
    </select>
    <script type="text/javascript">
    $(function() {
      $('#cl-rev-{{$cl.Number}}').change(function() {
        var r = $(this).val();
        var row = $('tr#cl-{{$cl.Number}}');
        row.addClass('pending');
        $.post('/assign', {
          'cl': '{{$cl.Number}}',
          'r': r
        }).success(function() {
          row.removeClass('pending');
          row.addClass('saved');
        }).error(function() {
          row.removeClass('pending');
          row.addClass('failed');
        });
      });
    });
    </script>
    {{end}}
    </td>
    <td>
      <a href="http://codereview.appspot.com/{{.Number}}/" title="{{ printf "%s" .Description}}">{{.Number}}: {{.FirstLineHTML}}</a>
      {{if and .LGTMs $tbl.Assignable}}<br /><span style="font-size: smaller;">LGTMs: {{.LGTMHTML}}</span>{{end}}
      {{if and .NotLGTMs $tbl.Assignable}}<br /><span style="font-size: smaller; color: #f74545;">NOT LGTMs: {{.NotLGTMHTML}}</span>{{end}}
      {{if .LastUpdateBy}}<br /><span style="font-size: smaller; color: #777777;">(<span title="{{.LastUpdateBy}}">{{.LastUpdateBy | shortemail}}</span>) {{.LastUpdate}}</span>{{end}}
    </td>
    <td title="Last modified">{{.ModifiedAgo}}</td>
    <td>{{if $.IsAdmin}}<a href="/update-cl?cl={{.Number}}" title="Update this CL">&#x27f3;</a>{{end}}</td>
  </tr>
{{end}}
{{else}}
<tr><td colspan="5"><em>none</em></td></tr>
{{end}}
{{end}}
</table>

<hr />
<address>
You are <span class="email">{{.User}}</span> &middot; <a href="{{.LogoutURL}}">logout</a><br />
datastore timing: {{range .Timing}} {{.}}{{end}}
</address>

  </body>
</html>
`))
