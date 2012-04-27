package dashboard

// This file handles the front page.

import (
	"bytes"
	"html/template"
	"io"
	"net/http"
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

func handleFront(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	data := &frontPageData{
		Reviewers: personList,
	}
	var currentPerson string
	currentPerson, data.UserIsReviewer = emailToPerson[user.Current(c).Email]

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

	if data.UserIsReviewer {
		tableFetch(0, func(tbl *clTable) error {
			q := activeCLs.Filter("Reviewer =", currentPerson).Limit(10)
			tbl.Title = "CLs assigned to you for review"
			tbl.Assignable = true
			_, err := q.GetAll(c, &tbl.CLs)
			return err
		})
	}

	tableFetch(1, func(tbl *clTable) error {
		q := activeCLs.Filter("Author =", currentPerson).Limit(10)
		tbl.Title = "CLs sent by you"
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
		if data.UserIsReviewer {
			for i := len(tbl.CLs) - 1; i >= 0; i-- {
				cl := tbl.CLs[i]
				if cl.Author == currentPerson || cl.Reviewer == currentPerson {
					tbl.CLs[i] = tbl.CLs[len(tbl.CLs)-1]
					tbl.CLs = tbl.CLs[:len(tbl.CLs)-1]
				}
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

	wg.Wait()

	select {
	case err := <-errc:
		c.Errorf("%v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	default:
	}

	var b bytes.Buffer
	if err := frontPage.ExecuteTemplate(&b, "front", data); err != nil {
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
      td {
        padding: 2px 5px;
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

{{range $tbl := .Tables}}
<h3>{{$tbl.Title}}</h3>
{{if .CLs}}
<table class="cls">
{{range $cl := .CLs}}
  <tr id="cl-{{$cl.Number}}">
    <td class="email">{{$cl.DisplayOwner}}</td>
    {{if $tbl.Assignable}}
    <td>
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
    </td>
    {{end}}
    <td>
      <a href="http://codereview.appspot.com/{{.Number}}/" title="{{ printf "%s" .Description}}">{{.Number}}: {{.FirstLineHTML}}</a>
      {{if and .LGTMs $tbl.Assignable}}<br /><span style="font-size: smaller;">LGTMs: {{.LGTMHTML}}{{end}}</span>
    </td>
    <td title="Last modified">{{.ModifiedAgo}}</td>
  </tr>
{{end}}
</table>
{{else}}
<em>none</em>
{{end}}
{{end}}

<hr />
<address>
datastore timing: {{range .Timing}} {{.}}{{end}}
</address>

  </body>
</html>
`))
